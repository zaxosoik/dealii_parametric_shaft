import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.colors as mcolors
from bs4 import BeautifulSoup
import meshio
import numpy as np
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
from scipy.interpolate import make_interp_spline
from scipy import interpolate
import xml.etree.ElementTree as ET
import glob

def parse_parameter(param):
    return {
        'value': param.find('value').text,
        'default_value': param.find('default_value').text,
        'documentation': param.find('documentation').text,
        'pattern': param.find('pattern').text,
        'pattern_description': param.find('pattern_description').text,
    }

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    parameters = {}

    for bearings in root.iter('Bearings'):
        for bearing in bearings:
            parameters[bearing.tag] = parse_parameter(bearing)

    return parameters

# Use the function
parameters = parse_xml('parameters.xml')
##print(parameters)
#with open('parameters.xml', 'r') as f:
#    data = f.read()

#soup=BeautifulSoup(data,features="html.parser")
#table=soup.find
##print(table)
# Open the HDF5 file
dir_path = r'solution_*.h5'
filenames = glob.glob(dir_path)
filenames.sort()
#print(filenames)

with h5py.File(filenames[-1], "r") as f: # type: ignore
    # Read datasets from the HDF5 file
    x_coordinates = (f['center'][:]).flatten()
    nodes = (f['nodes'][:])
    norm_of_stress = (f['norm_of_stress'][:]).flatten()
    cells = (f["cells"][:])
    
    #outer_shell_cells_check=(f['outer_shell_cells_check'][:]).flatten()
with h5py.File(filenames[0], "r") as f_1: # type: ignore
        delta_x = (f_1['delta_x'][:]).flatten()
        delta_y = (f_1['delta_y'][:]).flatten()
        delta_z = (f_1['delta_z'][:]).flatten()

for i in range(np.size(filenames)-1):
     with h5py.File(filenames[i], "r") as f_1: # type: ignore
        delta_x += (f_1['delta_x'][:]).flatten()
        delta_y += (f_1['delta_y'][:]).flatten()
        delta_z += (f_1['delta_z'][:]).flatten()
        
        
        
#with h5py.File("solution_0.500000.h5", "r") as f_1: # type: ignore
#    delta_x_1 = (f_1['delta_x'][:]).flatten()
#    delta_y_1 = (f_1['delta_y'][:]).flatten()
#    delta_z_1 = (f_1['delta_z'][:]).flatten()
#
#with h5py.File("solution_1.000000.h5", "r") as f_2: # type: ignore
#    delta_x_2 = (f_2['delta_x'][:]).flatten()
#    delta_y_2 = (f_2['delta_y'][:]).flatten()
#    delta_z_2 = (f_2['delta_z'][:]).flatten()    
    

    


mesh = meshio.read("boundary_id.vtu")
#with h5py.File("step_18_cyl_force_modal_v2/solution_boundary_ids_3.000000.h5", "r") as f_b:
#    # Read datasets from the HDF5 file
#    x_coordinates_b = (f['center'][:]).flatten()
#    nodes_b = (f['nodes'][:])
#    norm_of_stress_b = (f['norm_of_stress'][:]).flatten()
#    outer_shell_cells_check=(f['boundary_ids'][:]).flatten()

field_data = mesh.point_data
##print(delta_x.shape)
deformation=[]
for i in range(delta_x.shape[0]):
    deformation.append(np.sqrt(delta_x[i]**2+delta_y[i]**2+delta_z[i]**2))

#print("Deformation shape:",np.shape(deformation))
#print("Deformation:",np.max(deformation))
##print("Deformation:",deformation)
    
##print(field_data["boundary_id"])
##print(max(field_data["boundary_id"])
boundary_ids = field_data["boundary_id"]

boundary_ids_dot=[]
cell_boundary_index = []
count=0
count_shell=0
for i in range(int(np.size(boundary_ids,0)/24)):
    cell_boundary_index.append(0)
    for k in range(int(np.size(cells,1))):
        if boundary_ids[i*24+k]==2:
              boundary_ids_dot.append(boundary_ids[i+k])
              cell_boundary_index[i] = 2
              count_shell+=1
              break
        elif boundary_ids[24*i+k]==3:
                boundary_ids_dot.append(boundary_ids[i+k])
                cell_boundary_index[i] = 3
                count_shell+=1
                break
        count+=1

##print("Count of for-loop=",count," and outer shell cells=",count_shell)
##print(cell_boundary_index)   
#print("Cell Boundary index =",np.shape(cell_boundary_index)) 
#print("Boundary Id shape =",field_data["boundary_id"].shape)
#for i in range(int(boundary_ids_dot.shape[0]/4)): 
cells_outer_shell_index_shell=[]
cells_outer_shell_index_bearing=[]
cells_outer_shell_index = []
count=0
for i in range(np.size(cell_boundary_index)):
    if cell_boundary_index[i]==2:
        cells_outer_shell_index_shell.append(cells[i][:])
        cells_outer_shell_index.append(cells[i][:])
        count+=1
    elif cell_boundary_index[i]==3:
        cells_outer_shell_index_bearing.append(cells[i][:])
        cells_outer_shell_index.append(cells[i][:])
        count+=1
        
##print("Count of for-loop=",count)    
#print("Cells on outer shell:",np.shape(cells_outer_shell_index_shell))
#print("Cells on bearing shell:",np.shape(cells_outer_shell_index_bearing))

##print("Cells outer shell & bearing shell",cells_outer_shell_index_bearing, cells_outer_shell_index_shell)
points_outer_radius=[]

points_outer_shell_unique = np.unique(cells_outer_shell_index_shell)
##print("Unique points outer shellshell",points_outer_shell_unique.shape)
points_outer_bearing_unique = np.unique(cells_outer_shell_index_bearing)
points_outer_unique=points_outer_shell_unique

#cells_outer_shell_index = np.append(cells_outer_shell_index_shell,cells_outer_shell_index_bearing)
#print(np.shape(cells_outer_shell_index))
points_outer_unique= np.append(points_outer_unique,points_outer_bearing_unique)
#print("Unique points outer shell",points_outer_unique.shape)
unique_points_shell_index = np.unique(cells_outer_shell_index_shell,return_index=True)
unique_points_bearing_index = np.unique(cells_outer_shell_index_bearing,return_index=True)

##print(np.shape(unique_points_shell_index))
##print(points_outer_unique)
##print(unique_points_shell_index)
##print(np.shape(unique_points_bearing_index))

unique_points_index = np.append(unique_points_shell_index[1][:],unique_points_bearing_index[1][:])
#print(np.shape(unique_points_index))
#print(unique_points_index)
equal=[]
for i in range(points_outer_bearing_unique.shape[0]):
    if points_outer_bearing_unique[i]==unique_points_bearing_index[0][i]:
        equal.append(1)
#if points_outer_bearing_unique.shape[0]== np.size(equal):
    #print("True")

##print(np.max(unique_points_index))
##print(np.size(equal))
delta_y_unique = delta_y[points_outer_unique]
delta_z_unique = delta_z[points_outer_unique]
##print(delta_y_unique.shape)

for i in range(points_outer_unique.shape[0]):
    radius = np.sqrt((nodes[points_outer_unique[i],1]-delta_y_unique[i])**2+(nodes[points_outer_unique[i],2]-delta_z_unique[i])**2)
    #radius = np.sqrt((nodes[points_outer_unique[i],1]**2+nodes[points_outer_unique[i],2]**2) 
    points_outer_radius.append(radius)

points_outer_radius = np.array(points_outer_radius)
threshold = 0.30


mask = points_outer_radius>threshold

filtered_radii = points_outer_radius[mask]
filtered_unique_points_index = unique_points_index[mask]
##print(np.shape(points_outer_radius))
point_outer_index_final = []  
#cells_outer_shell_index_2d = np.array(cells).flatten()
filtered_radii = np.array(filtered_radii)
sorted_indices = np.argsort(filtered_radii,axis=0)[::-1]
point_outer_index_final = sorted_indices[:filtered_unique_points_index.shape[0]]
point_outer_index_final = point_outer_index_final.flatten()
point_outer_index_final = np.array(points_outer_unique)[point_outer_index_final]
##print(point_outer_index_final)

#print("point_outer_final shape:",np.shape(point_outer_index_final))

x_outer = nodes[point_outer_index_final,0]
y_outer = nodes[point_outer_index_final,1]-delta_y[point_outer_index_final]

z_outer = nodes[point_outer_index_final,2]-delta_z[point_outer_index_final]

norm_of_stress_outer = norm_of_stress[point_outer_index_final]

deformation_outer = np.array(deformation)[point_outer_index_final]
angle = np.arctan(z_outer/y_outer)

sorted_indices = np.argsort(angle,axis=0)
angle_ = angle[sorted_indices]
x_coordinates_outer = x_outer[sorted_indices]
norm_of_stress_outer_ = norm_of_stress_outer[sorted_indices]
deformation_outer_ = deformation_outer[sorted_indices]
y_max = []

x_max = []
z_max = []

angle_new = angle_
angle_temp = angle_[0]
angles_index=[0]
angles = [angle_[0]]
for i in range(angle_.shape[0]):
    if abs(angle_[i]-angle_temp)<0.1:
        angle_new[i]=(angle_temp)
        
    else:
        angle_temp = angle_[i]
        angle_new[i]=angle_temp
        angles.append(angle_temp)
        angles_index.append(i)
angles_index.append(angle_new.shape[0])      
angles.append(angle_[-1])  
angle_new = np.array(angle_new)    
angles = np.array(angles)
angles_index = np.array(angles_index)
#print('Angles index shape:',np.shape(angles_index))
#print('Angles index:',angles_index)

#sorted_index = np.argsort(x_coordinates_outer,axis=0)
#x_coordinates_outer_temp = x_coordinates_outer[sorted_index]
##angle_new_temp = angle_new[sorted_index]
#norm_of_stress_outer_temp = norm_of_stress_outer_[sorted_index]
#deformation_outer_temp = deformation_outer_[sorted_index]
##print(temp_x)
temp_z = -10000000
temp_y = angle_new[0]
temp_x = x_coordinates_outer[0]
temp_d = deformation_outer_[0]
deform2plot=[]


for k in range(angles_index.shape[0]-1):
    
    sorted_index = np.argsort(x_coordinates_outer[angles_index[k]:angles_index[k+1]],axis=0)
    x_coordinates_outer_temp = x_coordinates_outer[angles_index[k]:angles_index[k+1]]
    x_coordinates_outer_temp = x_coordinates_outer_temp[sorted_index]
    
    
    #angle_new_temp = angle_new[sorted_index]
    norm_of_stress_outer_temp = norm_of_stress_outer_[angles_index[k]:angles_index[k+1]]
    norm_of_stress_outer_temp = norm_of_stress_outer_temp[sorted_index]
    deflections_temp = deformation_outer_[angles_index[k]:angles_index[k+1]]
    deformation_outer_temp = deflections_temp[sorted_index]
    temp_z = -10000000
    temp_y = angles[k]
    temp_x = x_coordinates_outer_temp[0]
    temp_d = deformation_outer_temp[0]
    for i in range(angles_index[k+1]-angles_index[k]):
        if norm_of_stress_outer_temp[i]>temp_z: #or abs(angle[i]-temp_y )> 0.1:
            
                temp_y = angles[k]
                temp_z = norm_of_stress_outer_temp[i]
                temp_x = x_coordinates_outer_temp[i]
                temp_d = deformation_outer_temp[i]
                
        y_max.append(temp_y)
        x_max.append(temp_x)
        z_max.append(temp_z)
        deform2plot.append(temp_d)
    

max_stresses = np.array(z_max)
angles2plot = np.array(y_max)

x_coordinates2plot = np.array(x_max)
#x_coordinates2plot ,  angles2plot = np.meshgrid(x_coordinates2plot,angles2plot)
deform2plot = np.array(deform2plot)
##################
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


grid_x, grid_y = np.mgrid[min(x_coordinates2plot):max(x_coordinates2plot):1000j, min(angles2plot):max(angles2plot):1000j]

grid_z = griddata((x_coordinates2plot, angles2plot), max_stresses, (grid_x, grid_y), method='nearest')
grid_deform = griddata((x_coordinates2plot, angles2plot), deform2plot, (grid_x, grid_y), method='nearest')
norm = plt.Normalize(-grid_deform.max(), grid_deform.min())
#colors = cm.viridis(norm(grid_deform))
#rcount, ccount, _ = colors.shape



def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

#ax.scatter(x, y, z)



#cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ['red', 'yellow', 'green'])


#print("Norm of stress outer shape:",np.shape(norm_of_stress_outer))
#print("X outer shape:",np.shape(x_outer))

#print('Min of deform2plot=',grid_deform.min(),'Max of deform2plot=',grid_deform.max())
fig, ax_carpet = plt.subplots(subplot_kw={"projection": "3d"})
#fig = plt.figure()
m = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
m.set_array(-grid_deform)
#ax_carpet = fig.add_subplot(111, projection='3d')
ax_carpet.plot_surface(grid_x, grid_y, np.log10(grid_z), cmap='viridis')#rcount=rcount, ccount=ccount,facecolors=colors, shade=False)#, c= deform2plot,cmap='viridis')
ax_carpet.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax_carpet.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_carpet.set_xlabel('X')
ax_carpet.set_ylabel('Angle')
ax_carpet.set_zlabel('Norm of stress')

plt.colorbar(m,label="Deformation[m]")
#ax_carpet.set_box_aspect([20.0, 3.0, 3.0])
#set_axes_equal(ax_carpet)
ax_carpet.view_init(elev=18, azim=-152)
#ax_carpet.legend()
#plt.colorbar(surf)  # Optional, adds a colorbar to the plot
plt.savefig('Carpet_Outer_shell_stress_plot')
plt.savefig('Carpet_Outer_shell_stress_plot.png')
plt.savefig('Carpet_Outer_shell_stress_plot_transparent.png',transparent=True)
plt.savefig('Carpet_Outer_shell_stress_plot.eps',format="eps")
#plt.show()


        
        
#print('Shape of outer index final:',np.shape(point_outer_index_final))
#print(point_outer_index_final)
#print('Max radius=',np.max(filtered_radii))
#print('Min radius=',np.min(filtered_radii))
#print('Shape of outer points radius:',np.shape(points_outer_radius))
#print("Shape of x coordinates:", x_coordinates.shape[0])
#print("Shape of nodes",nodes.shape)
##print("Shape of norm_of_stress:", norm_of_stress.shape)
##print("Cells",cells)
##print("Shape of cell:", cells.shape)

#########
# Create a data frame with x-coordinates and corresponding norm_of_stress
data_stress = pd.DataFrame({
    'x_coordinates': x_coordinates,
    'norm_of_stress': norm_of_stress,
})

##print("Shape of deformation:", deformation)
deformation = np.array(deformation)
data_deform = pd.DataFrame({
    'x_coordinates': x_coordinates,
    'deformation': deformation,
})
 
# #print the first 5 rows of the data frame
#print(data_stress.head())

# Group by x-coordinate and find the maximum norm_of_stress for each group
data_grouped_max_stress = data_stress.groupby('x_coordinates').min().reset_index()
#data_grouped_max_stress = data.groupby(['x_coordinates','norm_of_stress']max().reset_index()
data_grouped_max_stress.columns = ['x_coordinates', 'max_norm_of_stress']#'outer_shell_cells_check']
#data_grouped_max_stress.sort_values('count', ascending=False).drop_duplicates(['x_coordinates','max_norm_of_stress'])
##print(max(outer_shell_cells_check))
x_coordinates_ = data_grouped_max_stress['x_coordinates']
norm_of_stress_ = data_grouped_max_stress['max_norm_of_stress']
##print(data_grouped_max_stress)
#length =abs( x_coordinates_[0] - x_coordinates_[-1] )
y_max = []

x_max = []
temp_y = 0
temp_x = x_coordinates_[0]
##print(temp_x)
for i in range(x_coordinates_.shape[0]):

    if norm_of_stress_[i]>temp_y or abs(x_coordinates_[i]-temp_x )> 0.2:
        
            temp_x = x_coordinates_[i]
            temp_y = norm_of_stress_[i]

    y_max.append(temp_y)
    x_max.append(temp_x)


max_stresses = y_max   
x_coord_max_stress = x_max    

data_grouped_max_deform = data_deform.groupby('x_coordinates').min().reset_index()
#data_grouped_max_deform = data.groupby(['x_coordinates','norm_of_stress']max().reset_index()
data_grouped_max_deform.columns = ['x_coordinates', 'deformation']#'outer_shell_cells_check']
#data_grouped_max_deform.sort_values('count', ascending=False).drop_duplicates(['x_coordinates','max_norm_of_stress'])
##print(max(outer_shell_cells_check))
x_coordinates__ = data_grouped_max_deform['x_coordinates']
deformation_ = data_grouped_max_deform['deformation']
#print(data_grouped_max_deform)
#length =abs( x_coordinates_[0] - x_coordinates_[-1] )
y_max_ = []

x_max_ = []
temp_y = -10000000
temp_x = x_coordinates__[0]
##print(temp_x)
for i in range(x_coordinates__.shape[0]):

    if abs(deformation_[i])>temp_y or abs(x_coordinates__[i]-temp_x )> 0.01:
    
            temp_x = x_coordinates__[i]
            temp_y = deformation_[i]
    
        
    y_max_.append(temp_y)
    x_max_.append(temp_x)


max_deform = np.array(y_max_)   
x_coord_max_deform = np.array(x_max_)
#print(np.shape(max_deform))

# Plot the maximum norm_of_stress for each x-coordinate
fig, ax1 = plt.subplots()

color = 'tab:blue'

ax1.set_xlabel('X Coordinate')
ax1.set_ylabel('Maximum Norm of Stress',color=color)
ax1.set_yscale('log')
ax1.plot(x_coord_max_stress,max_stresses,color=color)

ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Maximum Deformation of Structure',color=color)
ax2.plot(x_coord_max_deform, max_deform, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax1.set_title('Maximum Norm of Stress and Deformation for each X Coordinate')
fig.tight_layout()
plt.savefig('MaxStress_dx.png')
plt.savefig('MaxStress_dx_transparent.png',transparent=True)
plt.savefig('MaxStress_dx.eps',format="eps")
plt.show()
#plt.pause(5)
plt.close()



