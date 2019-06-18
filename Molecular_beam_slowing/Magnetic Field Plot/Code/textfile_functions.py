"""
Useful Functions for Files.
"""
import os
os.chdir("C:/Users/jp7de/OneDrive/Desktop/Molecular Beam Slowing/Numerical B_Field/code")


def convert_file(file_name, current, replace):
    """
    This function is designed to replace all current symbols, with the replace 
    values. Parameters current and replace can be either a list of a string.
    
    @type file_name: string
    @type current: List[str] || str
    @type replace: List[Str] || str
    @rtype: None
    """
    if type(current) is list and type(replace) is list:            
        if len(current) == len(replace):
            for i in range(len(current)):
                with open(file_name) as f: 
                    newText = f.read().replace(current[i], replace[i])
                with open(file_name, "r+") as f:
                    f.write(newText)
    else: 
        with open(file_name) as f:
            newText = f.read().replace(current, replace)
        with open(file_name, "r+") as f:
            f.write(newText)


def load_txtfile(file_name):
    """
    This function is used to load textfiles that may otherwise contain too large 
    amounts of data.
    
    The function reads the data in the textfile (Which should only contains 
    numbers) and returns all the numbers in a single list.
    
    @type file_name: str
    @rtype: str
    """
    list_values= []
    
    def remove(text, chars):
        """
        Removes all given chars in text.
        
        @type text: str
        @type char: str
        @rtype: str
        """
        for c in chars:
            text = text.replace(c, "")
        return text
            
    with open(file_name) as f:
        new_Text = f.readline()
        while new_Text != '':
            temp_list = remove(new_Text, "[],").split()
            for i in range(len(temp_list)):
                list_values.append(float(temp_list[i]))
            new_Text = f.readline()
    return list_values
        
        
def flatten_list(list,ignore_types=(str)): 
    """
    Takes a nested list of any dimension and recursively "flattens" it out 
    into a 1 dimension list.
    
    @type list: List Object
    @type ignore_types= (str): Ignores all string inputs
    @rtype: None
    """
    for item in list:
        if isinstance(item, Iterable) and not isinstance(item, ignore_types):
            yield from flatten_list(item,ignore_types=(str))
        else:
            yield item
            
            
def generator_to_list(gen):
    """
    Converts an abstract generator object into a list containing the same 
    elements as the generator.
    
    @type gen: Generator object
    @rtype: List[items]
    """
    temp_list = []
    for item in gen:
        temp_list.append(item)
    return temp_list
        
        
## Testing