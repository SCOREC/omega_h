cmake_minimum_required(VERSION 2.17)

set(CTEST_DO_SUBMIT ON)
set(CTEST_TEST_TYPE Nightly)
set(CTEST_BUILD_CONFIGURATION RelWithDebInfo)

set(CTEST_NIGHTLY_START_TIME "17:00:00 EST")
set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_SITE "checkers.scorec.rpi.edu" )
set(CTEST_DROP_LOCATION "/submit.php?project=SCOREC")
set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_BUILD_NAME  "linux-gcc-${CTEST_BUILD_CONFIGURATION}")

set(CTEST_DASHBOARD_ROOT "/lore/yus9/nightlyBuilds/omega_h_build/" )
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_FLAGS -j4)

set(CTEST_PROJECT_NAME "SCOREC")
set(CTEST_SOURCE_NAME src)
set(CTEST_BINARY_NAME build)

set(REPO_URL_BASE "https://github.com/SCOREC/omega_h")
set(BRANCH "master")
set(MERGE_AUTHOR "Nightly Bot <donotemail@scorec.rpi.edu>")

set(CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set(CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

set_property(GLOBAL PROPERTY SubProject "omega_h")
set_property(GLOBAL PROPERTY Label "omega_h")

set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION} "tmpnam")

if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file(MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif()
if(EXISTS "${CTEST_BINARY_DIRECTORY}")
  file(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}")
endif()
file(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")

find_program(CTEST_GIT_COMMAND NAMES git)
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

function(git_exec CMD ACTION)
  string(REPLACE " " ";" CMD2 "${CMD}")
  message("Running \"git ${CMD}\"")
  execute_process(COMMAND "${CTEST_GIT_COMMAND}" ${CMD2}
    WORKING_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/${CTEST_PROJECT_NAME}"
    RESULT_VARIABLE RETVAR)
  if(RETVAR)
    message(FATAL_ERROR "${ACTION} failed (code ${RETVAR})!")
  else()
    message("${ACTION} succeeded")
  endif()
endfunction(git_exec)

function(create_branch BRANCH_NAME TRACKING_NAME)
  git_exec("branch --track ${BRANCH_NAME} ${TRACKING_NAME}"
           "Creating branch ${BRANCH_NAME}")
endfunction(create_branch)

function(checkout_branch BRANCH_NAME)
  git_exec("checkout ${BRANCH_NAME}"
           "Checking out branch ${BRANCH_NAME}")
endfunction(checkout_branch)

function(setup_repo)
  if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/${CTEST_PROJECT_NAME}")
    message("Running \"git clone ${REPO_URL_BASE}.git ${CTEST_SOURCE_DIRECTORY}/${CTEST_PROJECT_NAME}\"")
    execute_process(COMMAND "${CTEST_GIT_COMMAND}" clone ${REPO_URL_BASE}.git
        "${CTEST_SOURCE_DIRECTORY}/${CTEST_PROJECT_NAME}"
        RESULT_VARIABLE CLONE_RET)
    if(CLONE_RET)
      message(FATAL_ERROR "Cloning ${REPO_URL_BASE}.git failed (code ${RETVAR})!")
    else()
      message("Cloning ${REPO_URL_BASE}.git succeeded")
    endif()
    # make local tracking versions of master branch
    if(NOT "${BRANCH}" STREQUAL "master")
      create_branch(${BRANCH} origin/${BRANCH})
    endif()
  endif()
endfunction(setup_repo)

function(check_current_branch BRANCH_NAME CONFIG_OPTS ERRVAR)
  file(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}/${BRANCH_NAME}")

  execute_process(COMMAND df -h /tmp OUTPUT_VARIABLE outVar)
  message(STATUS "df result {\n${outVar}}")
  execute_process(COMMAND pwd OUTPUT_VARIABLE outVar)
  message(STATUS "pwd result output {\n${outVar}}")

  ctest_configure(
      BUILD "${CTEST_BINARY_DIRECTORY}/${BRANCH_NAME}"
      SOURCE "${CTEST_SOURCE_DIRECTORY}/${CTEST_PROJECT_NAME}"
      OPTIONS "${CONFIG_OPTS}"
      RETURN_VALUE CONFIG_RET)
  if(CONFIG_RET)
    message(WARNING "${BRANCH_NAME} config failed (code ${CONFIG_RET})!")
  else()
    message("${BRANCH_NAME} config passed")
  endif()

  execute_process(COMMAND df -h /tmp OUTPUT_VARIABLE outVar)
  message(STATUS "df result {\n${outVar}}")
  execute_process(COMMAND pwd OUTPUT_VARIABLE outVar)
  message(STATUS "pwd result output {\n${outVar}}")

  ctest_build(
      BUILD "${CTEST_BINARY_DIRECTORY}/${BRANCH_NAME}"
      NUMBER_ERRORS NUM_BUILD_ERRORS
      NUMBER_WARNINGS NUM_BUILD_WARNINGS
      RETURN_VALUE BUILD_RET)
  if(NUM_BUILD_WARNINGS OR
      NUM_BUILD_ERRORS OR BUILD_RET)
    message(WARNING "
${BRANCH_NAME} build failed!
  ${NUM_BUILD_WARNINGS} warnings
  ${NUM_BUILD_ERRORS} errors
  code ${BUILD_RET}")
  else()
    message("${BRANCH_NAME} build passed")
  endif()

  ctest_test(
      BUILD "${CTEST_BINARY_DIRECTORY}/${BRANCH_NAME}"
      RETURN_VALUE TEST_RET)
  if(TEST_RET)
    message(WARNING "${BRANCH_NAME} testing failed (code ${TEST_RET})!")
  else()
    message("${BRANCH_NAME} testing passed")
  endif()

  if(CONFIG_RET OR
     NUM_BUILD_WARNINGS OR
     NUM_BUILD_ERRORS OR BUILD_RET OR
     TEST_RET)
    message(WARNING "some ${BRANCH_NAME} checks failed!")
    set(${ERRVAR} True PARENT_SCOPE)
  else()
    message("all ${BRANCH_NAME} checks passed")
    set(${ERRVAR} False PARENT_SCOPE)
  endif()

  if(CTEST_DO_SUBMIT)
    ctest_submit(PARTS Update Configure Build Test
        RETRY_COUNT 4
        RETRY_DELAY 30
        RETURN_VALUE SUBMIT_ERROR)
    if(SUBMIT_ERROR)
      message(WARNING "Could not submit ${BRANCH_NAME} results to CDash (code ${SUBMIT_ERROR})!")
    else()
      message("Submitted ${BRANCH_NAME} results to CDash")
    endif()
  endif()
endfunction(check_current_branch)

function(check_tracking_branch BRANCH_NAME CONFIG_OPTS ERRVAR)
  checkout_branch("${BRANCH_NAME}")
  ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}/${CTEST_PROJECT_NAME}"
      RETURN_VALUE NUM_UPDATES)
  if("${NUM_UPDATES}" EQUAL "-1")
    message(FATAL_ERROR "Could not update ${BRANCH_NAME} branch!")
  endif()
  message("Updated ${NUM_UPDATES} files")
  check_current_branch(${BRANCH_NAME} "${CONFIG_OPTS}" ERRVAL2)
  set(${ERRVAR} ${ERRVAL2} PARENT_SCOPE)
endfunction(check_tracking_branch)

ctest_start(${CTEST_TEST_TYPE})

if(CTEST_DO_SUBMIT)
  ctest_submit(FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
      RETURN_VALUE HAD_ERROR)
  if(HAD_ERROR)
    message(WARNING "Cannot submit SCOREC Project.xml!")
  endif()
endif()

SET(CONFIGURE_OPTIONS
  "-DOmega_h_USE_MPI=OFF"
  "-DOmega_h_USE_CUDA=ON"
  "-DOmega_h_CUDA_ARCH=80"
  "-DOmega_h_USE_Kokkos=ON"
  "-DKokkos_PREFIX=${CTEST_DASHBOARD_ROOT}/build-kokkos/install"
  "-DBUILD_TESTING=ON"
  "-DBUILD_SHARED_LIBS=OFF"
  "-DENABLE_CTEST_MEMPOOL=ON"
  "-DCMAKE_CXX_EXTENSIONS=OFF"
)

setup_repo()

check_tracking_branch("${BRANCH}" "${CONFIGURE_OPTIONS}" CHECK_ERR)
