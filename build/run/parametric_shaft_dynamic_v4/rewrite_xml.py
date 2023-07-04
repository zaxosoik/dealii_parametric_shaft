import xml.etree.ElementTree as ET

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

    def add_parameter(self, parameter):
        self.parameters.append(parameter)

class XMLHandler:
    def __init__(self, xml_data):
        self.tree = ET.ElementTree(ET.fromstring(xml_data))
        self.root = self.tree.getroot()
        self.components = []

    def parse(self):
        for component in self.root:
            comp = Component(component.tag, component)
            for parameter in component:
                name = parameter.tag
                value = parameter.find('value').text
                default_value = parameter.find('default_value').text
                documentation = parameter.find('documentation').text
                pattern = parameter.find('pattern').text
                pattern_description = parameter.find('pattern_description').text
                param = Parameter(name, value, default_value, documentation, pattern, pattern_description, parameter)
                comp.add_parameter(param)
            self.components.append(comp)

    def save(self, file_name):
        self.tree.write(file_name)

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
                        parameter.xml_element.find('value').text = new_value
                        return True
        return False

with open('parameters.xml', 'r') as f:
    xml_data = f.read()

handler = XMLHandler(xml_data)
handler.parse()

manager = ParameterManager(handler)

# Change the value of a specific parameter
if manager.set_parameter_value('Geometry', 'Half_20length', '15.'):
    print('Parameter value updated successfully.')
else:
    print('Failed to update parameter value.')

handler.save('parameters.xml')  # save the modified XML tree back to the file
