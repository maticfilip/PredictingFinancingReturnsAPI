#  API za Predikciju Financijskih Povrata

Ovaj projekt predstavlja FastAPI aplikaciju koja predviđa satni povrat financijskog instrumenta na temelju povijesnih OHLCTV podataka (Open, High, Low, Close, Trades, Volume).
Model koristi Random Forest regresiju, a aplikacija je pripremljena za izvođenje u Docker kontejneru radi lakše prenosivosti i testiranja.

--------------------------------------------------------------------------------

##  Opis projekta

Cilj projekta je razviti sustav koji:
- prima aktualne tržišne podatke (OHLCTV),
- računa financijske značajke (featuree),
- predviđa sljedeći satni povrat instrumenta,
- vraća predikciju putem REST API sučelja.

Model je treniran na satnim povijesnim podacima i koristi kombinaciju tehničkih indikatora, pokretnih prosjeka i volatilnosti za donošenje procjena.

--------------------------------------------------------------------------------

##  Tehnologije

- Python 3.10
- FastAPI — REST API framework
- scikit-learn — strojno učenje
- pandas, numpy — obrada podataka
- joblib — spremanje i učitavanje modela
- Docker — kontejnerizacija aplikacije
- Uvicorn — ASGI server za FastAPI

--------------------------------------------------------------------------------

##  Pokretanje putem Dockera

1.  docker build -t financial-api .

2. docker run -d -p 8000:8000 financial-api

3. http://localhost:8000/docs

--------------------------------------------------------------------------------
##  Primjer korištenja (POST /predict)

Ulazni JSON:

{
  "Date": "2025-10-23T13:00:00",
  "Open": 68500.0,
  "High": 68750.0,
  "Low": 68300.0,
  "Close": 68650.0,
  "Trades": 15400,
  "Volume": 250.3
}
--------------------------------------------------------------------------------

##  Endpointi

| Endpoint | Metoda | Opis |
|-----------|---------|------|
| /health   | GET     | Provjerava status API-ja |
| /info     | GET     | Vraća informacije o modelu i značajkama |
| /predict  | POST    | Prima ulazne podatke i vraća predikciju povrata |


