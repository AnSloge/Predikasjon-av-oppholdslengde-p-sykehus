
HVORDAN KJØRE KODEN


Før koden kan kjøres er det essensielt å laste ned og pakke ut programmet.



1. Åpne analyse.ipynb fil og trykk "Run All". Velg preferert kernel.

Programmet vil begynne å kjøre. Det kan ta noen minutter før alle modellene er trent ferdig.

I analyse.ipynb skal du kunne finne:

1. Visualiseringer: grafiske representasjoner av dataene og modellresultater.
2. Modeller: inkluderer resultater og RMSE
3. Prediksjoner: resultater lagres i predicions.csv. (merk at denne allerede er lagt ved i mappen).
4. Modellfil: model.pkl skal lagres i mappen. (merk at denne allerede er lagt ved i mappen).



2. For å kjøre applikasjonen anbefaler jeg å kjøre applikasjonen lokalt. Da gjør du følgende: 

Hvis du bruker windows/linux:

1. Åpne Windows Powershell.
2. Naviger deg til dit du lagrer programmet. cd /Users/ditt_brukernavn/Mappenavn
3. Kjøre appen på følgende vis: python app.py
4. Åpne din nettleser, og skriv inn følgende addresse: http://localhost:8080/

Hvis du bruker Mac:

1. Åpne Terminal
2. Naviger deg til dit du lagrer programmet. cd /Users/ditt_brukernavn/Mappenavn
3. Kjøre appen på følgende vis: python app.py
4. Åpne din nettleser, og skriv inn følgende addresse: http://localhost:8080/


For detaljert beskrivelse om hvordan man bruker applikasjonen/nettsiden, se "app_brukerdokumentasjon.txt". 


Hvis du følger disse trinnene vil du være i stand til å kjøre koden min og bruke applikasjonen for prediksjon av sykehusopphold. 