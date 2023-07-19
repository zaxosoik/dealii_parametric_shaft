import xml.etree.ElementTree as ET
import numpy as np
import shutil
import os
import subprocess


class Parameter:
    def __init__(self, name, value, default_value, documentation, pattern, pattern_description, xml_element):
        self.name = name
        self.value = value
        self.default_value = default_value
        self.documentation = documentation
        self.pattern = pattern
        self.pattern_description = pattern_description
        self.xml_element = xml_element

class Component:
    def __init__(self, name, xml_element):
        self.name = name
        self.xml_element = xml_element
        self.parameters = []
        self.subcomponents = []

    def add_parameter(self, parameter):
        self.parameters.append(parameter)

    def add_subcomponent(self, subcomponent):
        self.subcomponents.append(subcomponent)

class XMLHandler:
    def __init__(self, xml_data):
        self.tree = ET.ElementTree(ET.fromstring(xml_data))
        self.root = self.tree.getroot()
        self.components = []

    def parse(self):
        for component in self.root:
            comp = Component(component.tag, component)
            self.parse_parameters(comp, component)
            self.parse_subcomponents(comp, component)
            self.components.append(comp)

    def parse_parameters(self, parent_component, parent_element):
        for parameter in parent_element:
            if parameter.tag.startswith('value'):
                continue  # Skip the subcomponents section
            name = parameter.tag
            value_element = parameter.find('value')
            value = value_element.text if value_element is not None else None
            default_value_element = parameter.find('default_value')
            default_value = default_value_element.text if default_value_element is not None else None
            documentation_element = parameter.find('documentation')
            documentation = documentation_element.text if documentation_element is not None else None
            pattern_element = parameter.find('pattern')
            pattern = pattern_element.text if pattern_element is not None else None
            pattern_description_element = parameter.find('pattern_description')
            pattern_description = pattern_description_element.text if pattern_description_element is not None else None
            param = Parameter(name, value, default_value, documentation, pattern, pattern_description, parameter)
            parent_component.add_parameter(param)

    def parse_subcomponents(self, parent_component, parent_element):
        for subcomponent in parent_element:
            if not subcomponent.tag.startswith('value'):
                comp = Component(subcomponent.tag, subcomponent)
                self.parse_parameters(comp, subcomponent)
                self.parse_subcomponents(comp, subcomponent)
                parent_component.add_subcomponent(comp)

    def save(self, file_name):
        self.tree.write(file_name)

class ParameterManager:
    def __init__(self, handler):
        self.handler = handler

    def set_parameter_value(self, component_name, parameter_name, new_value):
        for component in self.handler.components:
            if component.name == component_name:
                if self.update_parameter_value(component, parameter_name, new_value):
                    return True
                for subcomponent in component.subcomponents:
                    if self.update_parameter_value(subcomponent, parameter_name, new_value):
                        return True
        return False

    def update_parameter_value(self, component, parameter_name, new_value):
        for parameter in component.parameters:
            if parameter.name == parameter_name:
                parameter.value = new_value
                if parameter.xml_element is not None:
                    value_element = parameter.xml_element.find('value')
                    if value_element is not None:
                        value_element.text = str(new_value)
                    return True
        return False


with open('batch_parameters.xml', 'r') as f:
    xml_data_batch = f.read()

handler_batch = XMLHandler(xml_data_batch)
handler_batch.parse()
manager_batch = ParameterManager(handler_batch)
#print(handler_batch.components)
subcomponents = []
count = 0
for component in handler_batch.components:
    print(f"Component: {component.name}")
    
    
    for subcomponent in component.subcomponents:
        subcomponents.append([0,0,0]) 
        print(f"  Subcomponent: {subcomponent.name}")
        param_count=0
        for parameter in subcomponent.parameters:
            
            if parameter.value != None:
                subcomponents[count][param_count]=float(parameter.value)
                param_count +=1
            
                print(f"    Parameter: {parameter.name}")
                print(f"      Value: {parameter.value}")
            
                    
                
        count +=1
print(count)    
print(subcomponents)
with open('parameters.xml', 'r') as f:
    xml_data = f.read()
handler = XMLHandler(xml_data)
handler.parse()
manager = ParameterManager(handler)
component_count = 0

#for component in handler.components:
#    component_count += 1
#    parameter_count = 0
#    for parameter in component.parameters:
#        parameter_count +=1
#        print(parameter_count)
#        if parameter.name==handler_batch.components[component_count].subcomponents[parameter_count].name:
#            print(handler_batch.components[component_count].subcomponents[parameter_count].parameters[0].name,handler_batch.components[component_count].subcomponents[parameter_count].parameters[0].value)

isExist = os.path.exists('batch_runs')
if isExist:
    print('Batch Runs Folder created successfully.')
else:
    os.makedirs('batch_runs')
isExist = os.path.exists('batch')
if isExist:
    print('Batch Output Folder created successfully.')
else:
    os.makedirs('batch')    
component_count = 0     
parameter_count = 0 
batch_count = 1       
subcomponents = np.array(subcomponents)
#floats = [float(x) for x in subcomponents]   
batch_names_list = []    
for component in handler.components:
    print(f"  Component: {component.name}")
    for parameter in component.parameters:
        print(f"    Parameter: {parameter.name}")
        if subcomponents[parameter_count][2]!=0:
            param_linspace = np.arange(float(subcomponents[parameter_count][0]),float(subcomponents[parameter_count][1]),float(subcomponents[parameter_count][2]))
            #print(param_linspace)
            for i in param_linspace:
                if manager.set_parameter_value(component.name, parameter.name,i):
                    print('Parameter:',parameter.name ,' value updated successfully to ', i)
                else:
                    print('Failed to update parameter value.')
                param_count_2 = 0
                for component_2 in handler.components:
                    for parameter_2 in component_2.parameters:
                        if param_count_2 != parameter_count:
                            if subcomponents[param_count_2][2]>np.abs(10-15):
                                param_linspace_2 = np.arange(float(subcomponents[param_count_2][0]),float(subcomponents[param_count_2][1]),float(subcomponents[param_count_2][2]))
                                for k in param_linspace_2:
                                    if manager.set_parameter_value(component_2.name, parameter_2.name,i):
                                        print('Parameter:',parameter_2.name ,' value updated successfully to ', k)
                                    else:
                                        print('Failed to update parameter value.')
                            
                                    batch_name = 'batch/parameters_'+str(batch_count).zfill(6)+'.xml'
                                    if manager.set_parameter_value('Solver','OutputFolder', 'batch_runs/'+str(batch_count).zfill(6)):
                                        print('Batch Output Folder updated successfully.')
                                    else:
                                        print('Failed to update Batch Output Folder.')
                                    batch_names_list.append(batch_name)
                                    handler.save(batch_name)
                                    isExist = os.path.exists('batch_runs/'+str(batch_count).zfill(6))
                                    if isExist:
                                        print('Batch Output Folder created successfully.')
                                    else:
                                        os.makedirs('batch_runs/'+str(batch_count).zfill(6))
                                    manager.set_parameter_value('Solver','postproscriptpath','batch_runs/'+str(batch_count).zfill(6)+'postpro_h5py_v2.py')
                                    shutil.copy('postpro_h5py_v2.py', 'batch_runs/'+str(batch_count).zfill(6))
                                    print('Batch Count: ', batch_count)
                                    batch_count +=1
                                    
             
                        param_count_2 +=1
        parameter_count += 1

def run_command_with_args(arg):
    command = "mpirun -np 12 ./parametric_shaft_dynamic_v4 " + arg
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error occurred: {stderr.decode()}")
    else:
        print(stdout.decode())
        
             
print('Batches Input File created: ',batch_count-1)        
print('----------------------------')
print('Batches Folders and Input Files Created, Running Batch Run...')
for i in batch_names_list:
    print('Running Batch number: ',i)
    run_command_with_args(i)    
# Change the value of a specific parameter
#if manager.set_parameter_value('Geometry', 'Half_20length',subcomponents[1][0]):
#    print('Parameter value updated successfully.')
#else:
#    print('Failed to update parameter value.')
#
#handler.save('parameters.xml')  # save the modified XML tree back to the file
