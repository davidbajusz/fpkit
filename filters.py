''' FPKit. Filtering rules implemented for interaction fingerprints. Note that the current implementation relies on the use of pandas, and - for filterResidues - Schrödinger-style column headers, such as: A523_charged
'''

def excludeBits(dataset, interaction_list):
    '''Exclude specific interaction types. This function will filter out any columns from *dataset*, whose header contains as a substring any strings from *interaction_list*. Requires column headers, without any limitations on their format.
    '''
    interactions=[column for column in dataset.columns if not any([i for i in interaction_list if i in column])]
    filtered=dataset.filter(items=interactions)
    
    return filtered

def filterInteractions(dataset):
    '''Filter for interactions with at least one occurrence. In other words, any interaction without a single occurrence in the whole *dataset* will be filtered out.
    '''
    interactions=[column for column in dataset.columns if 1 in dataset[column].values]
    filtered=dataset.filter(items=interactions)
    
    return filtered


def filterResidues(dataset):
    '''Filter for residues with at least one occurrence of any interaction.
    
    For identifying the residues, this requires column headers in the following general format: Axxx_yyyy, where *A* is the 1-character chain identifier *xxx* is the residue number and *yyyy* is the interaction type. (e.g. A523_charged)
    
    The residue number is read from the second character of the column header until the first underscore.
    '''
    residues=set([column.split('_')[0] for column in dataset.columns if 1 in dataset[column].values])
    dataframesPerResidue=[dataset.filter(regex=i+'_.*') for i in sorted(residues,key=lambda element: int(element.split('_')[0][1:]))]

    filtered=dataframesPerResidue[0]
    for i in dataframesPerResidue[1:]:
        filtered=filtered.join(i)
        
    return filtered
