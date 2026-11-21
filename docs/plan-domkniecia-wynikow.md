# Plan domknięcia wyników badania głównego oraz badań A i B

Stan na 21 listopada 2026 r. Celem jest przygotowanie pełnego materiału wynikowego do rozdziałów 6–8 pracy. Analizy należy wykonać na maszynie z oryginalnymi artefaktami, a do `results/final/` przekazać gotowe podsumowania CSV lub JSON. Wystarczą zapisane predykcje, rekordy dopasowań, manifesty i indeksy podziałów. Ponowny trening nie jest potrzebny.

## Stan materiałów

| Badanie | Dostępne wyniki | Do dostarczenia |
|---|---|---|
| Główne | 372 wyniki dla 132 par zbiór–metoda, metryki według klas, macierze pomyłek, wybrane konfiguracje i wagi zespołów, czasy, pokrycie i awarie | Bootstrap, pamięć podczas dopasowania, kontrola pierwotnych podziałów oraz zestawienie wyników walidacyjnych do porównania z testem |
| A – ważenie klas | 72 oceny, punkty pracy, krzywe, diagnostyka kontroli wyboru stanu i rang oraz przedziały dla kontrastów | Raport zgodności podziałów i wag, odtworzenie wyboru sześciu konfiguracji T |
| B – ablacje regularyzacji | 108 ocen, 864 rekordy foldów, 54 kontrasty z przedziałami, krzywe strat, luki, plateau i stosunki norm | Raport zgodności i zagnieżdżenia podzbiorów, odtworzenie rankingu OOF konfiguracji 21 |

Autor potwierdził zatwierdzenie reguł analizy przed uruchomieniem badania głównego i obu badań A/B. Potwierdził również wersje analiz i sumy kontrolne dostarczonych podsumowań A/B.

## Badanie główne

### `main_bootstrap.csv` — niepewność różnic na teście

Eksport powinien zawierać **276 rekordów**, po jednym na zbiór i porównanie.

- **36 kontrastów metod w warunku T**. RealMLP–CatBoost, TabM–CatBoost i ModernNCA–KNN na każdym z 12 zbiorów.
- **120 zmian D → T** oraz **120 zmian T → T+E**. Każda zmiana dotyczy tej samej metody i tego samego zbioru. TabPFN występuje wyłącznie w warunku D.
- W każdym rekordzie podać zbiór, typ porównania, metody i warunki po obu stronach, metrykę, obie wartości punktowe, różnicę, dolną i górną granicę przedziału 95%, liczbę replik nieokreślonych i status przedziału. Dodać identyfikatory przebiegów lub odwołanie do odpowiadających im rekordów `final_sources.csv`.

Dla zadań binarnych stosować ROC AUC, a dla Covertype, Diabetes 130-US i Helena stosować stratę logarytmiczną. Dodatnia różnica oznacza korzyść pierwszej metody w kontraście lub późniejszego warunku w porównaniu D → T lub T → T+E. Dla straty odwrócić znak względem różnicy wyników AUC. Sprawdzić, czy wartości punktowe obliczone z predykcji zgadzają się z `final_summary.csv`, gdzie `log_loss_val` zawiera stratę testową.

Użyć predykcji z wybranych przebiegów. Dla T+E odtworzyć kombinację prawdopodobieństw z zapisanych wag. W każdej replice ocenić obie strony na tych samych wylosowanych obserwacjach. Przyjąć 2000 replik, ziarno 42 i granice percentylowe 2,5% oraz 97,5%. Dla zadań dzielonych po wierszach losować ze zwracaniem w obrębie klas, zachowując ich liczności. Dla Diabetes 130-US losować całych pacjentów wraz z pobytami, bez warstwowania według etykiety pobytu. Zachować ustalone modele i wagi zespołu. Gdy choć jedna replika jest nieokreślona, pozostawić granice puste i podać liczbę takich replik. Przedziały opisują niepewność warunkową dla zapisanych modeli i podziału.

### `main_fit_memory.csv` — zużycie pamięci podczas dopasowania

Z oryginalnych `records/result.json` wyeksportować po jednym rekordzie dla każdej konfiguracji z przebiegów użytych w raporcie końcowym. Podać zbiór, metodę, identyfikator przebiegu, indeks konfiguracji, status, CPU/GPU oraz dostępność każdej miary. Zapisać szczyt RSS procesu i procesów potomnych oraz szczyt przydzielonej i zarezerwowanej pamięci PyTorch. Ujednolicić jednostkę, np. MiB, i oznaczyć brak odczytu osobno.

Połączenie z indeksami i wagami w `final_summary.csv` umożliwi zestawienie kosztu D, T i T+E. Dla T+E można podać maksimum dostępnych szczytów konfiguracji o dodatniej wadze wraz z pokryciem pomiarów. Taki wynik opisuje konkretne dopasowania. Czasy predykcji są już w końcowym eksporcie. [Definicje kosztów i zakres śledzenia pamięci](setup.md#report-costs-and-memory-boundaries)

### `main_split_audit.csv` i `main_validation_metrics.csv`

- **Kontrola podziałów.** Podać po jednym wierszu na zbiór z identyfikatorem podziału, licznościami części rozwojowej i testowej oraz wynikiem kontroli zgodności identyfikatorów i etykiet testowych między metodami. Podać liczbę wspólnych identyfikatorów między częścią rozwojową i testową, a dla Diabetes również liczbę wspólnych pacjentów. Sprawdzić pokrycie części rozwojowej przez wewnętrzną walidację oraz rozłączność treningu i walidacji w foldach tam, gdzie procedura ich używa. Dla grupowych foldów sprawdzić również rozłączność pacjentów. Wynik kontroli powinien wynikać z pierwotnych indeksów.
- **Metryki walidacyjne.** Podać zbiór, metodę, warunek D/T/T+E, metrykę, wartość OOF lub LOO i identyfikator przebiegu. Użyć zapisanych konfiguracji i wag. TabPFN nie ma metryki OOF ani LOO. Zestawienie pozwoli porównać zmianę jakości na walidacji ze zmianą na teście przy strojeniu i łączeniu konfiguracji.

## Badanie A — potwierdzenie wykonania i wyboru konfiguracji

Zachować istniejące CSV w `results/final/a-b-studies/`. Zawierają liczności klas, AP, AUC i stratę, progi, punkty pracy, diagnostykę rang, wariant kontroli doboru stanu, korektę logitu i przedziały. Sześć pustych przedziałów dla precyzji przy progu 0,5 wynika z reguły replik nieokreślonych. W czterech kontrastach nieokreślone były wszystkie 2000 replik, a w dwóch po pięć. Zachować puste granice i podać rzeczywiste liczby w opisie eksportu.

- **`study_a_audit.csv` lub JSON.** Dla każdej pary zbiór–metoda, wariantu i ziarna podać identyfikator przebiegu oraz wynik kontroli identyczności obserwacji testowych, etykiet i foldów. Potwierdzić zgodność foldów z przebiegiem źródłowym badania głównego, kod klasy rzadkiej oraz obliczenie wag wyłącznie z liczności treningowych każdego foldu. Wskazać brakujące lub niezgodne rekordy. Korektę logitu powiązać z predykcjami wariantu nieważonego.
- **`source-a.json` lub tabela OOF kandydatów.** Dla sześciu par podać źródłowy przebieg, indeks i wynik OOF wybranej konfiguracji, kryterium wyboru i rozstrzygnięcie remisów. Dołączyć wyniki OOF i statusy kandydatów albo raport potwierdzający kontrolę ich kompletu. Zestawienie pozwala odtworzyć wybór bez użycia wyników testowych.

## Badanie B — potwierdzenie podzbiorów i rankingu źródłowego

Zachować istniejące CSV w `results/final/a-b-studies/`. Zawierają wyniki wszystkich wariantów, trzy kontrasty AUC i straty wraz z efektami według ziaren, przedziały, budżety, krzywe, luki i normy. Zgodność liczności między wariantami sprawdzono z eksportu. Do kontroli tożsamości obserwacji potrzebne są pierwotne indeksy.

- **`study_b_subset_audit.csv` lub JSON.** Podać zbiór, fold, ziarno, udział treningu, liczności klas i identyfikatory przebiegów. Na podstawie indeksów sprawdzić zagnieżdżenie 25% ⊆ 50% ⊆ 100%, identyczność obserwacji między czterema wariantami i stałość walidacji oraz testu. W zestawieniu wystarczą wyniki kontroli, liczności i identyfikatory źródeł.
- **`source-b.json` lub tabela 26 kandydatów × 8 zbiorów źródłowych.** Podać wyniki OOF, rangi per zbiór i średnie rangi, indeksy i nastawy kandydatów, zastosowane kryterium dopuszczenia oraz statusy i budżety przebiegów. Wskazać osiem zbiorów i ich identyfikatory, udokumentować przyczynę wyłączenia Covertype oraz potwierdzić wynik konfiguracji 21. W razie rozbieżności opisać ją wraz z danymi źródłowymi.

[Procedury weryfikacji źródeł i artefaktów A/B](mechanism-studies.md)

## Pakiet do przekazania

Dołączyć krótki README lub manifest zawierający wersję skryptu analizy, tożsamość wejściowych eksportów lub przebiegów, definicje kolumn, jednostki, liczebność rekordów i sumy kontrolne nowych podsumowań. Każdego brakującego pomiaru, nieokreślonego przedziału lub niezrealizowanej kontroli dotyczy osobny jawny status. Przy niedostępności artefaktów odnotować zakres analizy możliwy do wykonania i brakującą podstawę wniosku.
