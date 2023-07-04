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
    xml_data = f.read()

handler = XMLHandler(xml_data)
handler.parse()

manager = ParameterManager(handler)

# Change the value of a specific parameter
if manager.set_parameter_value('Geometry', 'Half_20length_20Last', '20.'):
    print('Parameter value updated successfully.')
else:
    print('Failed to update parameter value.')

handler.save('batch_parameters.xml')  # save the modified XML tree back to the file
