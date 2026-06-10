# FAQ

## Tips

- Kom med noget ekstra til eksamen

### Spørgsmål

#### Skal man tale dansk? Spørgsmål er jo på engelsk?

Tal dansk. Selvom literature er engelsk. Fagtermer må selvfølgelig både være dansk og engelsk.

#### Snak mest udfra teori eller praksis (udgangspunkt i projekt)

50 % spørgsmålstid med fokus på projekt
50 % spørgsmålstid med fokus på teori (NOGET ANDET END DET FRA PROJEKTET)

- Eksempel: Projekt om GA. Spørgsmål om autoencoder eller whatever

#### Skal man gå i dybden med den del af projektet der IKKE er teori? Eksempelvis spil-logikken?

Nej. Vis det, men nej

#### Hvad nu hvis Mathias allerede har snakket om Emne X?

Skal man ikke forholde sig til.

#### Hvordan skal man svare på spørgsmål?

Ikke så meget ja / nej. "Løb gerne med bolden"

#### Må man bruger slide shows?

Ja. Det er anbefalet at bruge slides.
Det går sjældent godt når folk taler frit fra leveren.
Og det er svært at følge med som bedømmer, hvis man hopper mellem kode, jupyter notebook, et diagram OG TILBAGE IGEN
Hellere inkludere kode i slideshow end springe mellem slides og IDE. Udover VIM. VIM er gud.

#### Må man have noter med?

Ja - men anbefales ikke. Det er svært at tale frit med noter.
Hellere forberede sig godt og så kunne tale frit

##### Opfølgende spørgsmål. Noter til spørgsmålstiden?

Anbefales ikke.

#### Spørger I til syntaks og formler?

Ikke således at man skal huske en formel eller noget syntaks.
Der spørges om koncepter og forståelse.

#### Hvor vigtigt er det at kunne huske fagtermer / terminologi?

Det er vigtigere at kunne forklare koncepterne.
90 % er forståelse og 10 % er fagtermer.

Det er vigtigere at kunne forklare One-hot encoding end at kunne huske termet.

---

## Spørgsmål fra Henriks eksamensliste

### Nummer 4. What is the commmon use of this expression?

m er antaller af samples
1 / m = mean # Som alternativ til mode, median

Svaret er MSE (Mean Squared Error)

- _Vil ikke blive stilt til eksamen_

### Nummer 7. Can a latent space be interpreted directly?

Latent space er skjulte / abstrakte rum.
Vi ser dem i PSA encoding, autoencoders, GANs, VAEs, osv.
Hidden layers i neurale netværk er ikke latente rum. Det er bare læringslag.

OBS: Læs op latent space magic.

### Nummer 8. What is the data type of these examples?

- A monetary value
- Olympic medals
- Accuracy of a binary classification model

Monetary value: Ratio / interval data
Medaljer: Ordinal data
Præcision: Ratio data

Types:

| Parametrics Y/N | Name     | Description                                                                | Centralitet                                             |
| --------------- | -------- | -------------------------------------------------------------------------- | ------------------------------------------------------- |
| Yes             | Nominal  | Labels, kategorier uden orden                                              |                                                         |
| Yes             | Ordinal  | Labels, kategorier med orden                                               |                                                         |
| No              | Interval | Tal uden naturligt nulpunkt, hvor forskellen mellem værdier er meningsfuld | Mean, median, mode, range, variance, standard deviation |
| No              | Ratio    | Tal med naturligt nulpunkt, hvor forskellen mellem værdier er meningsfuld  | Mean, median, mode, range, variance, standard deviation |

### Nummer 10. Explain supervised / unsupervised learning

I supervised learning har vi et datasæt med labels.
Vi træner en model på det, og så kan den generalisere til nye data.
Helt konkret - Ved supervised kender vi Y, og ved unsupervised kender vi ikke Y.

Supervised:

- Random forrest
-

unsupervised:

- clustering
  - K-means
    - K-means er konveks, så den kan ikke løse two moons datasættet da det er konkavt
    - db-scan Kan løse two moons datasættet, da den ikke er konveks

### Nummer 15. What kind of problem are GA suited for?

Normalt ved supervised / unsupervised har vi et datasæt. Har vi ikke her.
Trial and error. Vi har en fitness funktion, og så prøver vi at optimere den.
Crossover and mutation. Vi har ikke en model, vi kan træne på data, så vi prøver os frem.

### Nummer 22. What is the purpose of mutation in GA?

Ved at inkludere mutation, så kan det være at vores agenter kommer ud i et område af søge-rummet,
som de ikke ville have fundet ved crossover alene.
