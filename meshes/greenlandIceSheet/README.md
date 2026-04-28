converted the netcdf3 formatted gis.exo to netcdf4 via the seacas `io_shell` utility to support hdf5 based sliced
reads via mpi-io (i.e., Omega_h_exodus.cpp read_sliced(...)).

```
(ins)smithc11@monopoly: ~/develop/omegahDev $ io_shell --netcdf4 ~/develop/omegahDev/omegahMaster/meshes/greenlandIceSheet/gis.exo ~/develop/omegahDev/omegahMaster/meshes/greenlandIceSheet/gis_nc4.exo                                                                                                                                                                                     
Input:    '/users/smithc11/develop/omegahDev/omegahMaster/meshes/greenlandIceSheet/gis.exo', Type: exodus
Output:   '/users/smithc11/develop/omegahDev/omegahMaster/meshes/greenlandIceSheet/gis_nc4.exo', Type: exodus
                                                                                               
                                                                                               
 Maximum Field size = 183,656 bytes (0.175 MiB) for field 'temperature'.                       
 Resize finished...                                                                            
                                                                                               
                                                                                               
 Input Region summary for rank 0:                                                              
 Number of Coordinates per Node =              2       
 Number of Nodes                =          2,087                                              
 Number of ElementBlocks        =              1                          
 Number of Elements             =          3,690                                               
 Number of NodeSets             =              1        Length of entity list =          2,087 
 Number of SideSets             =              1                                               
                                                                                               
 Number of time steps on database = 1                                                          
        Time step     1 at time 0.00000e+00                                                    
                                                                                               
 Output Region summary for rank 0:                                                                                                                                                             
 Database: /users/smithc11/develop/omegahDev/omegahMaster/meshes/greenlandIceSheet/gis_nc4.exo
 Mesh Type = Unstructured, Exodus. Change Sets = 0                                         
                                                                 Variables : Transient / Reduction
 Spatial dimensions =   2                                        Global     =   0         1
 Node blocks        =   1        Nodes         =   2,087         Nodal      =   5         0
 Edge blocks        =   0        Edges         =       0         Edge       =   0         0
 Face blocks        =   0        Faces         =       0         Face       =   0         0
 Element blocks     =   1        Elements      =   3,690         Element    =   2         0
 Structured blocks  =   0        Cells         =       0         Structured =   0         0                                                                                                    
 Node sets          =   1        Node list     =   2,087         Nodeset    =   0         0
 Edge sets          =   0        Edge list     =       0         Edgeset    =   0         0                                                                                                    
 Face sets          =   0        Face list     =       0         Faceset    =   0         0
 Element sets       =   0        Element list  =       0         Elementset =   0         0
 Element side sets  =   1        Element sides =     492         Sideset    =   0              
 Assemblies         =   0                                        Assembly   =   0         0    
 Blobs              =   0                                        Blob       =   0         0    
                                                                                               
 Time steps         =   1       (0 to 0)                                                       
                                                                                               
                                                                                               
        Total Execution Time = 0.04453 seconds.                                                                                                                                                
                                               
io_shell execution successful.    
```
