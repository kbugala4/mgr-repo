


Głębokie sieci neuronowe w zastosowaniu do klasyfikacji danych tabelarycznych

Celem pracy jest zbadanie skuteczności znanych z literatury metod zastosowania głębokich sieci neuronowych do tworzenia modeli klasyfikacji na podstawie danych tabelarycznych. W zakres pracy wchodzi przegląd literatury i wstępny wybór najbardziej obiecujących podejść oraz ich implementacja i eksperymentalna weryfikacja na wielu zbiorach danych z różnych dziedzin, zwłaszcza takich, dla których uzyskanie dobrej jakości predykcji jest trudne ze względu np. na niezrównoważenie klas, niewielką liczbę przykładów trenujących lub wysoką wymiarowość. W szczególności należy uwzględnić badanie skuteczności mechanizmów uwrażliwiania na rzadkie klasy i zwiększania odporności na nadmierne dopasowanie oraz porównanie z modelami tworzonymi za pomocą konwencjonalnych (nieneuronowych) algorytmów uczenia się. Do rozważenia jest także przystosowanie implementowanych metod do użycia w trybie aktywnego uczenia się i ocena ich przydatności do jego realizacji. 






W oparciu o przegląd przygotowany w “Deep Neural Networks and Tabular Data: A Survey”  i po zapoznaniu się z wybranymi referencjami, przygotowany został krótki artykuł wprowadzający do poruszanego problemu. Pozwala on zrozumieć istotę problemu i techniczne ograniczenia algorytmów głębokiego uczenia przy pracy z danymi tabelarycznymi. 
Oprócz wyjaśniania podstawowych pojęć, artykuł dokonuje również przeglądu literatury pochylającej się nad tym problemem. Przedstawione pozycje są doskonałym początkiem wdrożenia w przeprowadzone badania i przygotowane prace naukowe.


Cel pracy jest wybitnie badawczy i bardzo jasno (choć rozlegle) określony. Celem pracy jest weryfikacja przydatności algorytmów głębokiego uczenia do problemów klasyfikacji na podstawie danych tabelarycznych. Ocena skuteczności opierać się ma na porównaniu do wybranych algorytmów konwencjonalnych (tj. takich, które nie korzystają z metod neuronowych).

Największy nacisk położony zostanie na część badawczą. Ze względu na rozległość pojęć modeli opartych na uczeniu głębokim oraz algorytmów konwencjonalnych, wymagana jest bardzo wnikliwa analiza wielu podejść. Jedynie badanie na szeroką skalę pozwoli wybrać najlepszych “reprezentantów” każdej z grup algorytmów, na podstawie których ocena skuteczności może być rzetelnie wyprowadzona. Zakładane jest, że różne metody (preprocessing danych, wybór architektury modelu, mechanizmy w sieci) mogą przynosić równie różne wyniki: od bardzo niskiej skuteczności, po uzyskanie (potencjalnie) wybitnie wytrenowanych klasyfikatorów.

Złożoność części technicznej, oprócz implementacji środowiska testowego, zależeć będzie w głównej mierze od wyboru znalezionych w literaturze modeli. Precyzując, nie każdy opisany w pracach naukowych algorytm może być dostępny w otwartoźródłowej formie gotowej klasy, modelu, “paczki” lub biblioteki - w takim przypadku wymagana będzie implementacja takiego modułu lub jego niedostępnego fragmentu

