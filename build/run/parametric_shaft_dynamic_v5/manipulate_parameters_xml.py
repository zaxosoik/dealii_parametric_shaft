import xml.etree.ElementTree as ET

class Parameter:
    def __init__(self, name, value, default_value, documentation, pattern, pattern_description):
        self.name = name
        self.value = value
        self.default_value = default_value
        self.documentation = documentation
        self.pattern = pattern
        self.pattern_description = pattern_description

class Component:
    def __init__(self, name):
        self.name = name
        self.parameters = []

    def add_parameter(self, parameter):
        self.parameters.append(parameter)

class XMLHandler:
    def __init__(self, xml_data):
        self.tree = ET.ElementTree(ET.fromstring(xml_data))
        self.root = self.tree.getroot()
        self.components = []

    def parse(self):
        for component in self.root:
            comp = Component(component.tag)
            for parameter in component:
                name = parameter.tag
                value = parameter.find('value').text
                default_value = parameter.find('default_value').text
                documentation = parameter.find('documentation').text
                pattern = parameter.find('pattern').text
                pattern_description = parameter.find('pattern_description').text
                param = Parameter(name, value, default_value, documentation, pattern, pattern_description)
                comp.add_parameter(param)
            self.components.append(comp)

class ParameterManipulator:
    def __init__(self, component):
        self.component = component

    def change_parameter_value(self, parameter_name, new_value):
        for parameter in self.component.parameters:
            if parameter.name == parameter_name:
                parameter.value = new_value
                print(f"Value of parameter '{parameter_name}' changed to '{new_value}'")

class ParameterManager:
    def __init__(self, handler):
        self.handler = handler

    def set_parameter_value(self, component_name, parameter_name, new_value):
        # Iterate over all components
        for component in self.handler.components:
            if component.name == component_name:
                # If the component is found, iterate over all parameters
                for parameter in component.parameters:
                    if parameter.name == parameter_name:
                        # If the parameter is found, update its value
                        parameter.value = new_value
                        return True
        return False

with open('parameters.xml', 'r') as f:
    xml_data = f.read()

handler = XMLHandler(xml_data) # Pass the XML data string, not the filename
handler.parse()

# Now you can access the parameters like this:
for component in handler.components:
    print(f"Component: {component.name}")
    for parameter in component.parameters:
        print(f"  Parameter: {parameter.name}")
        print(f"    Value: {parameter.value}")
        print(f"    Default value: {parameter.default_value}")
        print(f"    Documentation: {parameter.documentation}")
        print(f"    Pattern: {parameter.pattern}")
        print(f"    Pattern description: {parameter.pattern_description}")

# Manipulate the values of parameters
manipulator = ParameterManipulator(handler.components[2]) # Pass the first component
manipulator.change_parameter_value('Half_20length', '20') # Change the value of parameter named 'parameter1' to 'new value'
manager = ParameterManager(handler)

# Change the value of a specific parameter
if manager.set_parameter_value('Geometry', 'Half_20length', '20'):
    print('Parameter value updated successfully.')
else:
    print('Failed to update parameter value.')