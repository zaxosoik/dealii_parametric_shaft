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

xml_data = """YOUR XML DATA HERE"""
handler = XMLHandler(xml_data)
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
