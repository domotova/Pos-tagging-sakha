# Pos-tagging-sakha
## Актуальность задачи.
В данной работе рассмотрена задача морфологического анализа - частеречная разметка якутских текстов. Под этим понимается этап автоматического определения частей речи слов в тексте.

Сложность разработки объясняется тем, что нет хороших (размеченных) данных для обучения, а также якутский язык сам по себе имеет сложную структуру.
___
### План работы:
Якутский язык и его части речи.
1. Сбор датасета.
    * Анализ существующих решений для малоресурсных языков
    * Данные 
    * Очистить данные
    * Разметить данные
2. Модель машинного обучения. Его архитектура.
3. Анализ полученных результатов
4. Выводы
___
### Якутский язык и его части речи.
__Якутский язык относится к тюркской группе языков и обладает следующими свойствами:__
* Агглютинация. Словообразование происходит за счет добавления к основе аффиксов
* Аналитические приемы образования слов: сложение основ
* Гармония гласных: гласные звуки следуют друг за другом в строго определенном порядке.

Следуя фонетическим правилам к основе может быть добавлено множество аффиксов. В результате количество форм у одного слова может доходить до нескольких тысяч.

__Части речи якутского языка. Особенности разметки__

Разметка ведется в формате CoNLL-U (URL: https://universaldependencies.org/format.html). Каждому слову задается метка части речи. В данной работе метками являются следующие:
*	NOUN - аат тыл (существительное)
*	ADJ - даҕааһын аат (имя прилагательное)
*	NUM - ахсаан аат (имя числительное)
*	PRON - солбуйар аат (местоимение)
*	VERB - туохтуур (глагол) 
*	ADV - сыһыат (наречие) 
*	PR - дьөһүөл (?)
*	CONJ - ситим тыл (союз) 
*	PART - эбиискэ (частица)
*  AUX - сыһыан тыл (?) 
*	INTJ - саҥа аллайыы (междометие) 
*	PUNCT - пунктуация 
*  SYM - символлар
*  X - атын (другое)
___
### Сбор датасета.
__Анализ существующих решений для малоресурсных языков:__
 * Всего нашла 4 статьи:
   * Статья-1. (URL: https://arxiv.org/pdf/1904.05426.pdf), где текст разбивают на несколько кластеров (unsupervised learning), потом каждому кластеру присваивают тег. Точность примерно составляет 50%.
   * Статья-2. (URL: https://arxiv.org/pdf/1906.02656.pdf)
   * Статья-3. (URL: https://www.aclweb.org/anthology/2020.emnlp-main.391.pdf)
   * Статья-4. (URL: https://wlv.openrepository.com/bitstream/handle/2436/623727/Bolucu_Can_A_Cascaded_Unsupervised_Model_for_PoS_Tagging.pdf?sequence=3&isAllowed=y)
 * Решение, аналогично как у Георгия Петрова, с задачей NER (Named Entity Recognition). Представлено тут: https://github.com/georgiypetrov/ner-sakha. Работает для задачи POS-tag плохо, результат разметки можете увидеть в папке data файл transl_sakha_with_tag.txt
 * Ручная разметка. Минимум разметить 10000 предложений!!! Но это дорого и трудоемко)
 
__Данные.__
Данные взяты:
 * из репозитория sakha-embeddings (URL: https://github.com/nlp-sakha/sakha-embeddings). Это корпус, состоящий из статей якутской Википедии и электронных новостных изданий;
 * корпус3 дал научный руководитель. 
 
__Очистка данных.__
Очищаем от всех ненужных символов. Приводим буквы к единому кодированию символов. Код есть в папке colab_notebooks -- clean_data.ipynb

__Разметка данных.__
На данный момент собран корпус, состоящий из 2380 предложений, вручную размеченных по частям речи. Большую благодарность выражаю студентам ИЯКН СВФУ. Корпус находится в папке data с названием postag_sakha.conllu.
___
### Модель машинного обучения. Его архитектура.

Выбираем простую модель сверточной нейронной сети, построенную по идеям статьи Yu X., Faleńska A., Vu N. T. A general-purpose tagger with convolutional neural networks //arXiv preprint arXiv:1706.01723. – 2017 и использующую skip connections как в нейросети ResNet.

Обучение модели происходит с перекрестной энтропией (cross-entropy) в качестве целевой функции со скорость обучения 0,005 на 1700 предложениях корпуса

Оптимальное число эпох равна 26, размеры мини-пакетов выбраны равными 64 предложениям.

Код модели находится в папке colab_notebooks -- cnn_postag_sakha.ipynb.
___
### Анализ полученных результатов.
Качество работы алгоритмов классификации на несбалансированных данных традиционно оценивается с помощью точности (precision), полноты (recall) и среднего гармонического точности и полноты классификации (f1-score) по классам. Ниже в таблице приведены точность, полнота и f1-мера нейросети для каждого класса тестовых данных, а в последнем столбце их количество.

|  Tag  | precision  | recall | f1-score | quantity |
| NOUN  |  0,82      |  0,9   |   0,86   |  3082    |
| ADJ   |  0,65      |  0,62  |   0,63   |   605    |
| NUM   |  0,91      |  0,86  |   0,88   |   281    |
| PRON  |  0,95      |  0,88  |   0,91   |   471    |
| VERB  |  0,83      |  0,83  |   0,83   |   1812   |
| ADV   |  0,59      |  0,38  |   0,46   |   346    |
| PART  |  0,56      |  0,43  |   0,49   |   276    |
| PR    |  0,59      |  0,48  |   0,53   |   233    |
| AUX   |  0,32      |  0,29  |   0,3    |    63    |
| CONJ  |  0,62      |  0,61  |   0,62   |   170    |
| INTJ  |  0,25      |  0,4   |   0,31   |    5     |

Доля верных ответов обученной сверточной нейронной сети равна 78 % на тестовых данных и 84% на обучающих данных. 
___
### Выводы.

* сформирован корпус из 2380 предложений с частеречной разметкой, который опубликован в общераспространенном открытом доступе тут.
* в тестовом наборе данных получена точность 78% при классификации членов предложения по частям речи сверточными нейросетями. Это позволяет использовать разработанную нейросеть для первичной обработки данных для последующего ручного исправления и верификации разметки.
* в ходе исследования были замечены уникальные аффиксы, такие как -ааччы и др., присущие только глаголам или иным частям речи. Это позволяет в дальнейшем организовать правила выявления частей речи.

В дальнейшем будем улучшать точность!
___
