import bpy
import pandas as pd

url = 'https://raw.githubusercontent.com/virtualdvid/MachineLearning/master/csv_files/iris.csv'
df = pd.read_csv(url)
        
scale_factor = 0.1
for i in range(len(df)):
     
    v = df.iloc[i]
     
    if v[4] == 'setosa':
		bpy.ops.mesh.primitive_cone_add(location=(0,0,0))
    if v[4] == 'versicolor':
        bpy.ops.mesh.primitive_cube_add(location=(0,0,0))
    if v[4] == 'virginica':
        bpy.ops.mesh.primitive_uv_sphere_add(location=(0,0,0))
     
    bpy.context.object.name = 'row-' + str(i)
     
    bpy.ops.transform.resize(value=(v[3]**0.1*scale_factor,)*3)
    bpy.ops.transform.translate(value=(v[0], v[1], v[2]))