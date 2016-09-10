import pandas as pd
import re
import six
import string


def load_place_names(
        location='../datasets/placenames.csv', kinds=None,
        countries=None, reduce_char_set=True):

    """ Dataset can be retrieved from:

    https://developers.google.com/adwords/api/docs/appendix/geotargeting

    Parameters
    ----------
    location: str
        Path to dataset.
    kinds: str or list of str (optional)
        If supplied, only rows whose "Target Type" entry is in
        this list will be returned.
    countries: str or list of str (optional)
        If supplied, only rows whose "Country Code" entry is in
        this list will be returned.
    reduce_char_set: boolean
        Whether to reduce the character set to a-z and underscore, replacing
        with the underscore standing in for all non a-z characters.

    Returns
    -------
    place_names, characters

    """
    df = pd.read_csv(location)

    if kinds is not None:
        if isinstance(kinds, str):
            kinds = [kinds]

        col = 'Target Type'
        include = df[col] == kinds[0]
        for kind in kinds[1:]:
            include = include | (df[col] == kind)
        df = df[include]

    if isinstance(countries, str):
        countries = [countries]
    if isinstance(countries, list):
        countries = set(countries)

    place_names = {}
    for country_code, group in df.groupby('Country Code'):
        if countries is None or (country_code in countries):
            place_names[country_code] = list(group['Name'])

    if reduce_char_set:
        for country_code, pns in six.iteritems(place_names):
            pns = (pn.lower() for pn in pns)
            pns = [re.sub('[^a-z]', '_', pn) for pn in pns]
            place_names[country_code] = pns
        characters = set(string.ascii_lowercase + '_')
    else:
        characters = set()
        for country_code, pns in six.iteritems(place_names):
            for pn in pns:
                characters |= set(pn)

    return place_names, list(characters)


def to_numeric(names, characters):
    assert isinstance(characters, list) or isinstance(characters, str)
    inv_char = {c: i for i, c in enumerate(characters)}

    numeric_names = []
    for name in names:
        numeric_names.append([inv_char[c] for c in name])
    return numeric_names


if __name__ == "__main__":
    place_names, characters = load_place_names(
        kinds='City', countries='CA', reduce_char_set=True)
    numeric_names = {
        cc: to_numeric(pns, characters)
        for cc, pns in six.iteritems(place_names)}
