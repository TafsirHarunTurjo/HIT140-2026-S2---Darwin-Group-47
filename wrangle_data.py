
import pandas as pd 

read_data = "raw_data.csv"
players = pd.read_csv(read_data)
# store all the coloumns in dataset named as players coloumn
players.columns = [c.strip() for c in players.columns]
# Keeping only the required coloumns from the raw data 
required_cols = ["Player", "Pos", "MP", "Min"]
players = players[[c for c in required_cols if c in players.columns]].copy()
# if any row has any missing info so we can drop that row 
players = players.dropna(subset=["Player", "Pos", "MP", "Min"])
players["Min"] = players["Min"].astype(str).str.replace(",", "").astype(float)
print(players.columns.tolist())
players["MP"] = pd.to_numeric(players["MP"])
# if any player doesn't played any match
players = players[players["MP"] > 0]
# Keeping only fowrard and defense foword player we will not include any player with multiple positions
players = players[players["Pos"].isin(["FW", "DF"])]
players.to_csv("cleaned.csv", index=False, encoding="utf-8-sig")

print("cleaned dataset saved as 'cleaned.csv'")
print(players.groupby("Pos")["Min"].describe())
