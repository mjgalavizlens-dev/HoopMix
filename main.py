import pandas as pd

# Cargar los datos del CSV
df = pd.read_csv("partidos.csv")

print("--- Jugadores con más de 25 puntos ---")
# Filtramos la tabla manteniendo solo las filas donde Puntos sea mayor que 25
top_anotadores = df[df["Puntos"] > 25]
print(top_anotadores)

print("\n--- Jugadores de los Lakers ---")
# Filtramos solo los jugadores del equipo Lakers
lakers = df[df["Equipo"] == "Lakers"]
print(lakers)