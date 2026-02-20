import geopandas as gpd
import pycountry

# Percorso shapefile originale
shapefile_path = "../images/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"
# Percorso shapefile modificato
output_path = "../images/ne_50m_admin_0_countries/world_fixed.shp"

# Leggi shapefile
world = gpd.read_file(shapefile_path)

# Funzione per trovare ISO3 da nome
def name_to_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

# Trova paesi con nome ma ISO_A3 mancante o -99
missing_iso = world[(world['NAME'].notna()) & ((world['ISO_A3'].isna()) | (world['ISO_A3'] == '-99'))]

print("Paesi con nome ma senza codice ISO_A3 valido:")
for idx, row in missing_iso.iterrows():
    iso3 = name_to_iso3(row['NAME'])
    print(f"{idx+1}. {row['NAME']} -> vecchio ISO_A3: {row['ISO_A3']} -> nuovo ISO_A3: {iso3}")
    # aggiorna shapefile
    world.at[idx, 'ISO_A3'] = iso3 if iso3 else row['ISO_A3']

# Salva shapefile corretto
world.to_file(output_path)

print(f"\nShapefile aggiornato salvato in: {output_path}")
