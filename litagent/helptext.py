"""Встроенная справка: `python -m litagent help [тема]`. Тексты — здесь, чтобы не расходиться с кодом (см. тест)."""
from __future__ import annotations

import difflib

TOPICS: dict[str, tuple[str, str]] = {
    "start": ("с чего начать", """\
1. Задайте e-mail (нужен для вежливых запросов): LITAGENT_EMAIL или `email` в litagent.toml (образец: litagent.example.toml).
2. Проверьте доступность источников и модели:   python -m litagent check
3. Ручной поиск:  python -m litagent manual -q "crispr; sickle cell" --oa-only
4. Авто-поиск:    python -m litagent auto "цель на человеческом языке" --notes literature_search_systems.tex
Подробно: help manual | auto | presets | sites | import | pmc | pdf | speed | clean | llm | web | files | tags | supp | limits"""),
    "manual": ("ручной режим", """\
python -m litagent manual -q "тема" [-q "ещё тема"] [--topics файл.txt]
Мини-язык темы: `;` — И между группами, `|` — ИЛИ внутри группы, `-слово` — исключение.
  --sources a,b      какие источники (встроенные и свои)     --limit N          записей с источника
  --site имя         искать ТОЛЬКО на одном сайте            --pick-site        выбрать один сайт из списка по номеру
  --no-pmc           PubMed без поиска в PMC и без скачивания оттуда
  --pdf-only         только статьи с реально доступным полным PDF (см. help pdf)
  --workers N        параллельных скачиваний (по умолчанию 10; 1 — по одному; см. help speed)
  --year-from/--year-to   диапазон лет                        --oa-only          только открытый доступ
  --title-only       искать по названию                       --strict           каждая группа обязана встретиться
  --max-downloads N  потолок скачиваний (0 — без потолка)     --dry-run          показать запросы, ничего не отправляя
  --supplements      докачать доп. материалы статей (см. help supp)
  --auto-tags        теги скачанных статей по их ключевым словам (см. help tags)
  -n, --count N      скачать N статей с PDF: поиск углубляется, пока не наберётся N (см. help pdf)
  --deepen-rounds K  сколько раз можно углублять поиск при --count (по умолчанию 5)
  --preset имя       взять параметры из пресета (см. help presets)   --save-preset имя   сохранить параметры как пресет
  --flat             не раскладывать файлы по папкам источников (см. help files)"""),
    "presets": ("пресеты ручного режима", """\
Сохранить сложный запрос и запускать его одним именем (файл ./presets.json, настройка presets_file или --presets-file):
  python -m litagent manual -q "graph; neural|gnn; -survey" --year-from 2020 --sources pubmed,openalex -n 30 --save-preset gnn
  (с --dry-run параметры только сохраняются, ничего не отправляется; --preset-desc "текст" — описание)
  python -m litagent manual --preset gnn                   запуск; -q из командной строки ДОБАВЛЯЮТСЯ к темам пресета
  python -m litagent manual --preset gnn --year-from 2023  значения и --sources/--site/--count из командной строки главнее
  python -m litagent preset list | show имя | remove имя [-y]
В пресет попадают темы, группы, исключения, --raw, категории arXiv, источники, годы, --limit, режимы скачивания; e-mail, ключи и пути — нет.
Файл — обычный JSON, его можно править вручную."""),
    "auto": ("авто-режим (локальная нейросеть)", """\
python -m litagent auto "цель" [--notes файл.tex|.md|.txt] [--llm-filter] [--rounds N] [--dry-run]
--site имя / --pick-site: поиск только на одном сайте; нейросеть тогда видит в контексте только его и пишет запрос в его синтаксисе.
Модель строит план запросов; код проверяет его (запрещённые источники отбрасываются). Свои сайты из custom_sites.json
попадают в контекст модели автоматически. --dry-run показывает план и запросы без поиска.
Нет модели/ответ мусорный — агент переходит на эвристику. Что видит модель из заметок: python -m litagent notes файл.tex"""),
    "sites": ("свои сайты и опросник", """\
python -m litagent sites add | edit [имя] | list | show [имя] | test [имя] | remove [имя]
  add/edit   опросник «вопрос-ответ»: адрес API, параметр запроса, синтаксис, фильтры, страницы, ключ, пути в JSON
             Enter — принять значение в [скобках], «-» — очистить поле, «/q» — отмена.
  test       реальный запрос «machine learning» и проверка разбора полей (--query "..." — своя тема)
Файл: custom_sites.json (--sites-file или `custom_sites` в litagent.toml). Подходят только JSON-API по GET.
Дальше: help import"""),
    "import": ("загрузка списка сайтов из интернета", """\
python -m litagent sites import <https://…/sites.json | путь/к/файлу.json> [--only имена] [--overwrite] [--enable] [-y]
Выбор: после списка спросит «Что добавить?» — Enter = все, либо номера/имена (1,3 · 2-4 · name1,name2), «нет» = отмена.
  --only a,b  заранее выбрать конкретные;  -y без --only = добавить все без вопросов.
Экономно: ОДИН запрос, не больше 256 КБ, без повторов, паузы по хосту; до 50 сайтов за раз.
Безопасно (файл из сети недоверенный):
  * сначала показывается нумерованный список (имя, хост, статус), запись — только после вашего выбора (или -y);
  * ключи из файла вырезаются — при необходимости настройте их сами: sites edit имя;
  * тексты чистятся и режутся (они попадают в промпт модели); сайты приходят ВЫКЛЮЧЕННЫМИ (включить: --enable
    или sites edit); пауза между запросами — не меньше 1 с;
  * адрес файла и адреса сайтов проверяются политикой (https, блок-лист); существующие имена не перезаписываются без --overwrite.
Формат файла — как custom_sites.json: {"sites": [ {...}, ... ]} (образец: custom_sites.example.json). Для GitHub нужна ссылка Raw."""),
    "pmc": ("PubMed + PMC: поиск и PDF", """\
PubMed ищет сразу в двух базах одним запросом: PubMed (аннотации) и PMC (полнотекстовый архив, E-utilities db=pmc); записи
объединяются по PMID/DOI. PDF скачиваются только официальным путём: OA Web Service (oa.fcgi) -> ссылка на PDF или пакет
статьи на ftp.ncbi.nlm.nih.gov (по https); из пакета берётся основной PDF. Только для статей из Open Access Subset.
Сайт PMC (pmc.ncbi.nlm.nih.gov, /pmc/articles/…/pdf) агенту по-прежнему закрыт: массовая загрузка оттуда запрещена.
Отключить: --no-pmc или pubmed_pmc = false в litagent.toml. Лицензия: отдельные PDF из oa_pdf — для некоммерческого использования."""),
    "pdf": ("только статьи с полным PDF", """\
python -m litagent manual -q "тема" --pdf-only [--max-downloads N]      (то же в auto; в litagent.toml: pdf_only = true)
Режим включает --oa-only (источники сразу сужают поиск) и считает статью найденной, только если PDF РЕАЛЬНО скачан и проверен
(не лендинг и не paywall). XML-копии (Europe PMC) не берутся. Сначала проверяются записи, где PDF вероятен (ссылка из источника,
PMCID, is_oa); попыток до 10×max_downloads. В results.csv — только статьи с PDF; отброшенные перечислены в run.json (pdf_only).
Проверка = скачивание, поэтому с --no-download режим несовместим.
-n/--count N: «скачать ровно N статей с PDF» (включает --pdf-only и --max-downloads N). Если кандидаты кончились, а N не набрано,
агент повторяет поиск с вдвое большим лимитом на источник (до deepen_limit_cap = 1000), пропуская уже проверенные записи,
и так не более --deepen-rounds раз (по умолчанию 5). Работает в manual и auto. Итог — в run.json → target."""),
    "speed": ("ускорение скачивания", """\
--workers N (litagent.toml: download_workers) — параллельные скачивания, по умолчанию 10. К одному хосту: не чаще min_interval и
одно соединение за раз; к разным хостам — одновременно. Для файлов: download_retries = 2, download_timeout = 25 с, Retry-After
дольше 20 с -> кандидат пропускается; хост с 3 сбоями подряд пропускается до конца запуска; заглушка без %PDF обрывается сразу.
Поиск тоже параллельный: источники опрашиваются одновременно (до 10 потоков), паузы по хостам сохраняются.
Фильтр релевантности LLM (--llm-filter) оценивает пакеты статей в 4 параллельных запроса к модели.
Если файлы в основном на одном хосте, ускорения не будет (это правило вежливости). В конце печатается время скачивания."""),
    "clean": ("очистка папки со статьями", """\
python -m litagent clean [--articles] [--cache] [--logs] [--all] [--dry-run] [-y] [--out папка]
Без параметров удаляет ВСЁ: статьи, HTTP-кэш (.http_cache/) и логи (отчёты запусков runs/). Параметры выбирают только нужное:
  --articles  .pdf/.xml (и .part) из pdf/ и xml/ (включая папки источников), чужие файлы не трогает; пустые папки источников убирает
  --cache     HTTP-кэш: повторные запросы снова пойдут в сеть     --logs (или --runs)  runs/<время>/run.json и results.csv
  --all       то же, что без параметров. Индекс приводится в порядок. Без -y спрашивает подтверждение; --dry-run только показывает."""),
    "llm": ("модель для авто-режима", """\
По умолчанию: Ollama, модель gemma4, адрес http://127.0.0.1:11434   (ollama pull gemma4)
  --model имя  --llm-url адрес  --llm-provider ollama|openai (LM Studio, llama.cpp: http://localhost:1234/v1)
GigaChat (облако Сбера, локальная модель не нужна):
  [keys] gigachat_client_id + gigachat_client_secret  # Client ID и Client Secret из кабинета Sber Studio
  (или переменные GIGACHAT_CLIENT_ID / GIGACHAT_CLIENT_SECRET; Base64-ключ агент соберёт сам)
  python -m litagent auto "цель" --llm-provider gigachat [--model GigaChat-2-Pro]
  Адрес https://api.giga.chat/v1; токен агент получает сам (OAuth, живёт 30 минут, обновляется).
  Scope: --gigachat-scope GIGACHAT_API_PERS (физлица, по умолчанию) | GIGACHAT_API_CORP | GIGACHAT_API_B2B.
  Нужен сертификат Минцифры: установите Russian Trusted Root CA в систему либо укажите PEM-файл
  в [llm] ca_bundle / --gigachat-ca-bundle (скачать: https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt).
Те же настройки — в разделе [llm] файла litagent.toml. Проверка связи: python -m litagent check"""),
    "web": ("веб-интерфейс в браузере", """\
python -m litagent web [--host АДРЕС] [--port N] [--no-browser] [--config файл]
По умолчанию http://127.0.0.1:8765 (только этот компьютер), браузер открывается сам. Настройки — [web] в litagent.toml.
Разделы: Поиск (ручной и авто-режим, лог задачи вживую, отмена, таблица результатов), Библиотека (PDF открываются в браузере),
Запуски (отчёты runs/: запросы, ошибки, раунды плана LLM), Пресеты (запуск и удаление), Сайты (вкл/выкл своих сайтов),
Конфиг (редактор litagent.toml: почта, ключи, LLM, скачивание; ошибки показываются под полями, файл создаётся при сохранении).
Поиск запускается через тот же CLI, что и консоль — поведение одинаковое. Одна задача за раз.
Безопасность: в конфиге лежат ключи — не выставляйте сервер в сеть (--host 0.0.0.0), если не понимаете, кто может подключиться."""),
    "tags": ("авто-теги по ключевым словам", """\
--auto-tags (litagent.toml: auto_tags = true) — каждой скачанной статье проставляются теги по её ключевым словам.
Ключевые слова берутся из самих источников: keyword-поля EuropePMC и OpenAlex (topics/keywords/MeSH-major), категории arXiv,
поле fields.keywords у своих сайтов. Поскольку такие поля приходят мусорными строками («Mechanisms, Crispr/cas; Grna | Cas
Classification»), каждая фраза разбивается по разделителям (; | , и перевод строки), нормализуется, отсеивается стоп-лист
(служебные и демографические термины) и шум; если keyword-поля пусты — теги собираются из заголовка (капитализированные
термины и n-граммы без вложенных дублей) и повторяющихся слов аннотации.
Ручные теги не перетираются; теги хранятся в index.json (поле tags) и не зависят от файлов.
В веб-интерфейсе (Библиотека): «⚙ авто» на карточке и «⚙ Авто-теги немеченым» наверху (статьи, у которых тегов ещё нет).
API: POST /api/library/tags/auto {pk} или {all:true} (опционально force:true)."""),
    "supp": ("дополнительные материалы статей", """\
--supplements (litagent.toml: supplements = true) — вместе со статьёй качать приложения/данные/медиа:
  * пакет PMC Open Access (.tar.gz): всё содержимое, кроме основного PDF и разметки (.nxml/.xml/.sgm);
  * Europe PMC supplementaryFiles (.zip) по PMCID;
  * свои сайты: supp_flag (признак «есть доп. материалы»), supp_path (список ссылок в записи) и
    supp_url+id_param (шаблон пакета, {id}/{doi}) — настраиваются в sites add/edit, форме «Сайты» и custom_sites.json.
Файлы складываются в library/supp/<источник>/<имя статьи>/ и регистрируются в index.json (поле supp); zip/tgz распаковываются.
Лимиты: файл до 40 МБ, всё вместе до 200 МБ и до 40 файлов на статью; имена из архивов плоские (path traversal невозможен).
В веб-интерфейсе (Библиотека) кнопка «↓ докачать доп. материалы» показывается только там, где материалы есть (по данным
источника) — либо после «проверить наличие» (дешёвый lite-запрос Europe PMC, кэшируется). Там же ручная докачка по URL. API:
  POST /api/library/supplements/fetch {pk}, /check {pk}, /url {pk,url}, /delete {pk,file}.
Удаление статьи удаляет и её доп. материалы; clean --articles чистит supp/ целиком."""),
    "files": ("какие файлы создаются", """\
library/pdf/<источник>/, library/xml/<источник>/   скачанные статьи по папкам сайтов-источников (openalex, pubmed, имя вашего сайта…)
library/supp/<источник>/<статья>/   доп. материалы (--supplements; см. help supp)
python -m litagent organize [--dry-run]   разложить по папкам уже скачанное; --flat или folders_by_source = false — не раскладывать
library/index.json   что уже скачано (повторно не качается)
library/runs/<время>/      run.json (запросы, ошибки) и results.csv (статусы)       library/.http_cache  кэш ответов
custom_sites.json          свои сайты (создаётся при sites add / sites import)       litagent.toml        ваши настройки"""),
    "limits": ("что агент соблюдает всегда", """\
Только легальные источники и открытые копии (Unpaywall/OA). Sci-Hub, Google Scholar и т.п. заблокированы политикой
(policies.json). Паузы по хосту, Retry-After, 401/403 = остановка источника. Свои сайты подчиняются тем же правилам."""),
}


def render(topic: str | None = None) -> str:
    if not topic:
        w = max(map(len, TOPICS))
        rows = "\n".join(f"  {k:<{w}}  {v[0]}" for k, v in TOPICS.items())
        return f"Справка: python -m litagent help <тема>\n\n{rows}\n\nКоманды: manual, auto, check, notes, sites, clean, web, help (у каждой есть -h)."
    t = topic.strip().lower()
    if t in TOPICS:
        return f"== {t}: {TOPICS[t][0]} ==\n{TOPICS[t][1]}"
    near = difflib.get_close_matches(t, TOPICS, n=2)
    return f"Нет темы «{topic}»." + (f" Возможно: {', '.join(near)}." if near else "") + "\n\n" + render()
