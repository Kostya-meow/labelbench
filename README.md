# LabelBench — локальная авторазметка изображений

**Платформа экспериментов:** датасеты с GT без копирования фото, confidence каждой
модели, polygon NMS / consensus, precision / recall / F1, история и отмена.
Свой метод из notebooks можно оценить через импорт JSON или API.

Инструкция: [docs/experiments.md](docs/experiments.md).
Исследования и план сравнений: [docs/research-2026.md](docs/research-2026.md).

**Устойчивость и объединение границ:** отдельный фоновый worker, SQLite checkpoints,
9 преобразований без сохранения копий фото, medoid/голосование масок, абляции семейств,
живой дашборд, отмена/продолжение и статистический отчёт.
[Протокол и научные основания](docs/robustness-experiment.md).

Локальный API и браузерный интерфейс для сравнения предсказаний семейств
моделей: **PP-OCRv5 Server**, **PP-OCRv6 Medium Det**, **Mask2Former**, **SAM 2.1**, **YOLO26-seg**, **RF-DETR Historical Textline**,
**Doc-UFCN Generic Historical Line**, **Eynollah Textline** и **Riksarkivet RTMDet Lines**.
Новые модели добавляются
одним адаптером, не меняя API и интерфейс.

## Что умеет

- читает изображения из `data/images/`;
- сохраняет каждый запуск отдельно: `data/output/<provider>/<run-id>/`;
- хранит унифицированный COCO-подобный JSON с боксами, полигонами/RLE-масками,
  метками, confidence и происхождением;
- показывает исходник, наложения моделей и предварительный consensus score;
- после inference позволяет для каждого provider-а включать/выключать результат, выбирать цвет каждого класса сегмента и скачать оставленные аннотации отдельным JSON;
- отправляет выбранные результаты вместе с изображением в локальный VLM через LM Studio и показывает улучшенную разметку отдельным слоем;
- работает без загруженных ML-пакетов: интерфейс запускается, а статус сразу
  объясняет, какой provider недоступен и почему;
- загружает веса при первом обращении в стандартный кэш Hugging Face/PaddleOCR.

## Быстрый старт (Windows)

Нужны Python 3.10, Git и `uv` (`winget install Astral-sh.uv`). Последовательно запустите:

1. `bat\INSTALL_GPU.bat`
2. `bat\DOWNLOAD_WEIGHTS.bat`
3. `bat\START.bat`

Установщик создаёт отдельные окружения `.venv-gpu` и `.venv-ocr`, чтобы CUDA-библиотеки
PyTorch и Paddle не конфликтовали в одном Windows-процессе.

Откройте <http://127.0.0.1:8000>, скопируйте изображения в `data/images/` и нажмите
«Запустить выбранные». Веса заранее скачивает `bat\DOWNLOAD_WEIGHTS.bat`.

Если остальные модели уже установлены, для PP-OCRv6 достаточно закрыть сервер,
запустить `bat\INSTALL_PPOCR6.bat`, затем `bat\DOWNLOAD_PPOCR6.bat` и `bat\START.bat`.
PP-OCRv6 использует PyTorch CUDA в `.venv-gpu`; окружение PaddleOCR не меняется.
Обе версии PP-OCR возвращают только детекцию и полигоны, без распознавания текста.

### Сохранённая разметка

PP-OCRv6 доступен в трёх вариантах: `ppocr6` (Medium), `ppocr6_small` и
`ppocr6_tiny`. Все возвращают детекцию текстовых строк и полигоны без распознавания
текста. Установка общая: `bat/INSTALL_PPOCR6.bat`; загрузка всех трёх —
`bat/DOWNLOAD_PPOCR6.bat`. Веса Small и Tiny:
[Small](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_safetensors),
[Tiny](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_safetensors).

Повторный запуск использует кэш детекторов в `data/output/cache`: изображение
сравнивается по SHA-256, учитываются модель, параметры, локальные веса и код адаптера.
В карточке появляется «Из кэша». Для нового inference включите
«Пересчитать, не использовать сохранённое». Пустые результаты тоже кэшируются.
Старые результаты из `data/output/<provider>/<run>/result.json` подхватываются
по имени модели и изображения, размеру изображения и времени изменения файла.
Они помечаются «Сохранено (старый запуск)»: старые файлы не содержат контрольной
суммы и настроек, поэтому при сомнении используйте принудительный пересчёт.

### Проверка разметки через LM Studio или RouterAI

У детекторов и VLM есть отдельные журналы с прошедшим временем и этапами,
которые поступают с сервера во время обработки. В журнале видны очередь,
использование кэша, inference, отправка частей и повтор ответа.
Это этапы работы приложения, а не вывод внутренних рассуждений модели.
Кнопка «Остановить VLM» закрывает текущий HTTP-запрос и запрещает следующие части
и повторы. Исходная разметка сохраняется; после остановки можно отправить новый запрос.
Уже выполненная работа внешнего API может тарифицироваться: закрытие соединения
не гарантирует возврат средств или остановку вычислений на стороне провайдера.
`402 Insufficient balance` означает отказ API из-за недостатка средств:
ответа модели нет, исходная разметка сохранена. Пополните баланс или выберите LM Studio.

В поле «Сервис» выберите **RouterAI**, вставьте API-ключ и точное имя vision-модели,
например `deepseek/deepseek-v4-pro-0813`. Изображение отправляется в base64 через
`https://routerai.ru/api/v1/chat/completions`; ключ не сохраняется в файлы,
localStorage или экспорт разметки. После закрытия вкладки его нужно ввести снова.
LM Studio остаётся отдельным локальным вариантом.

Для обоих сервисов доступны «По частям» и «Все сразу». Во втором режиме все
выбранные координаты передаются одним запросом без скрытого разделения при ошибке
контекста. Лимит токенов ответа можно менять отдельно (512–32768).
Неполный или некорректный ответ автоматически повторяется один раз.
RouterAI использует обычный chat API и тот же валидатор решений; JSON-схема
принудительной генерации используется в LM Studio.

В LM Studio запустите локальный сервер на `http://localhost:1234` и загрузите
**qwen/qwen3-vl-4b** — эта модель проверена через интерфейс с изображением.
LabelBench предпочитает Qwen VL при первом заполнении списка моделей.
После обычного inference оставьте включёнными нужные слои provider-ов,
задайте промпт и нажмите «Отправить в VLM». В запросе передаются
исходное изображение и компактные полигоны результатов этих моделей. Каждый запрос
независимый: предыдущие JSON не пересылаются, чтобы экономить контекст.
Фильтры отдельных классов управляют отрисовкой и экспортом; в VLM передаются все
классы включённых provider-ов. Начните с одного детектора текстовых строк.

В VLM передаются только нормализованные в 0–1 сегментационные полигоны до 50 точек;
боксы не передаются. Перекрывающиеся кандидаты заранее объединены в группы.
`pick` выбирает ровно один вариант группы, `fuse` задаёт единую итоговую границу,
`drop` удаляет группу, `uncertain` оставляет её на ручную проверку. Необработанная группа
с дублями никогда автоматически не считается верной.
Экспорт содержит компактных кандидатов и исходные решения модели для аудита.
В режиме «По частям» результаты отправляются порциями до 40 групп и примерно 16 000 символов
координат в одной части. Кандидаты одной группы не разделяются. При переполнении
контекста часть автоматически делится ещё раз. Если сервер перестал отвечать,
успешные части сохраняются, остальные отмечаются сомнительными; интерфейс сообщает
о частичном ответе. Противоречивые решения повторяются один раз и не принимаются
автоматически. В LM Studio ответ ограничен JSON-схемой: один ключ на группу и один допустимый
выбор. Для Qwen3-VL-4B можно использовать контекст 32768.
Проверка по частям выбирает и корректирует существующие сегменты; автоматическое
добавление пропусков в этом режиме отключено, чтобы модель не принимала объекты
из соседней части за отсутствующие.

Под ответом автоматически появляется отдельное изображение с итоговыми полигонами VLM
и количеством объектов по статусам. Зелёный — верно, красный пунктир — удалено,
жёлтый — сомнительно, синий — исправлено, фиолетовый — объединено, бирюзовый — добавлено.
Каждый слой можно скрыть. Галочка над ответом показывает только принятые объекты на верхнем изображении.
Оборванный JSON повторяется один раз с короткой инструкцией и не считается
успешной проверкой. Ответ VLM требует проверки человеком: модель может ошибаться в координатах.

Если модель видна в списке, но при отправке появляется `Failed to load model`,
загрузите её кнопкой загрузки модели в LM Studio и дождитесь окончания загрузки.
Список в `/v1/models` подтверждает наличие модели, но не гарантирует, что она
уже загружена в VRAM.

Для проверки изображения используйте vision-модель, например `Qwen3-VL-4B`.
Если LM Studio пишет `does not support image inputs`, текущая конфигурация модели
не принимает изображения. Сам файл `mmproj` нельзя загружать как основную модель:
он должен работать вместе с соответствующими основными весами и поддерживаемым runtime.

> **SAM 2.1:** для GPU-инференса Meta рекомендует WSL2/Linux; скрипт всё равно
> поддерживает Windows, но CUDA extension может не собраться. Базовый inference
> обычно продолжает работать без extension, с ограниченным post-processing.

## Самостоятельная установка через BAT

Для ручной установки без фоновых задач в папке `bat/` есть три файла:

1. `INSTALL_GPU.bat` — создаёт окружение и проверяет работающую CUDA у PyTorch и Paddle.
2. `DOWNLOAD_WEIGHTS.bat` — скачивает веса всех моделей только после успешной GPU-проверки.
3. `START.bat` — поднимает интерфейс на `http://127.0.0.1:8000`.

Нужны установленный `uv`, Git (для SAM 2) и доступ к официальным индексам PyTorch,
PaddlePaddle, GitHub, Hugging Face и Paddle model hosting. Каждый BAT можно
повторно запускать после обрыва сети.

## Providers и модели по умолчанию

| Provider | Роль | Вес по умолчанию |
| --- | --- | --- |
| `ppocr` | детекция текстовых областей и polygons, без OCR-текста | PP-OCRv5 Server detection (PaddleOCR 3.x) |
| `ppocr6` | детекция текста и polygons через Transformers/PyTorch CUDA, без OCR-текста | [`PaddlePaddle/PP-OCRv6_medium_det_safetensors`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_safetensors) |
| `mask2former` | panoptic/instance-кандидаты классов COCO | `facebook/mask2former-swin-large-coco-panoptic` |
| `sam2` | class-agnostic object masks | `facebook/sam2.1-hiera-large` |
| `yolo26` | COCO instance segmentation | `yolo26n-seg.pt` |
| `rfdetr_historical` | только historical text lines; регионы отбрасываются до обработки масок | `Kansallisarkisto/rfdetr_textline_textregion_detection_model` |
| `docufcn` | generic historical text lines | `Teklia/doc-ufcn-generic-historical-line` |
| `eynollah_textline` | historical textline segmentation | [`SBB/eynollah-textline`](https://huggingface.co/SBB/eynollah-textline) |
| `rtmdet_lines` | instance segmentation исторических текстовых строк | [`Riksarkivet/rtmdet_lines`](https://huggingface.co/Riksarkivet/rtmdet_lines) |

Для исторического **Mask R-CNN** отдельный публичный checkpoint с воспроизводимым inference API
не найден, поэтому COCO/обычный scene-text checkpoint сюда не подставлен.
У **APAU-Net** опубликованы training notebooks, но публичного checkpoint/ONNX нет; provider будет
добавлен, когда авторы опубликуют веса, пригодные для inference.

Настройки задаются переменными окружения: `LABELBENCH_DEVICE` (`auto`, `cpu`,
`cuda`), `LABELBENCH_PPOCR6_MODEL`, `LABELBENCH_MASK2FORMER_MODEL`, `LABELBENCH_SAM2_MODEL`,
`LABELBENCH_SAM2_POINTS_PER_SIDE`, `LABELBENCH_YOLO26_MODEL`,
`LABELBENCH_RFDETR_CHECKPOINT`, `LABELBENCH_DOCUFCN_CHECKPOINT`,
`LABELBENCH_RTMDET_CHECKPOINT`, `LABELBENCH_RTMDET_CONFIG`, `LABELBENCH_RTMDET_PYTHON`,
`LABELBENCH_EYNOLLAH_CHECKPOINT`, `LABELBENCH_EYNOLLAH_PYTHON`.
Для LM Studio доступны `LABELBENCH_LM_STUDIO_URL` (по умолчанию
`http://localhost:1234/v1`), `LABELBENCH_LM_STUDIO_API_KEY` и
`LABELBENCH_LM_STUDIO_TIMEOUT`.

## API

- `GET /api/health` — доступность providers.
- `GET /api/images` — список входных изображений.
- `POST /api/runs` — запустить одно изображение: `{"image_name":"a.jpg","providers":["ppocr","sam2"]}`.
- Для PP-OCRv6 передайте `"providers":["ppocr6"]` в тот же endpoint.
- `GET /api/runs/{run_id}` — объединённый результат.
- `GET /api/runs/{run_id}/overlay/{provider}` — наложение provider-а.
- `GET /api/llm/models` — список моделей, опубликованных LM Studio.
- `POST /api/llm/review` — отправить изображение и выбранные результаты в VLM;
  ответ содержит `refined_annotations`, `notes` и исходный ответ модели.

## Как добавить ещё одну модель

Создайте класс в `src/labelbench/providers/`, наследующий `AnnotationProvider`,
и зарегистрируйте его в `registry.py`. Он возвращает `ProviderResult` и поэтому
сразу становится доступен в API, в папке output и в UI. Детали — в
[`docs/adding_provider.md`](docs/adding_provider.md).

## Автоматическая разметка: как интерпретировать результат

PP-OCR и Mask2Former делают независимые предсказания. SAM 2.1 формирует
автоматические proposal-маски сеткой point prompts; это не семантические классы.
`consensus_score` — пространственное согласие с другими кандидатами, а не
истинная точность. Результаты предназначены для ускорения проверки человеком,
а не для автоматического объявления ground truth.

## Проверка интерфейса

`node --test tests/overlay.test.cjs` выполняет реальный `app.js` и проверяет
отрисовку всех providers, скрытие, цвет и переключение на результат VLM.
Регрессия пустого canvas была вызвана обращением к `state.llmApply.checked`
вместо DOM-элемента `elements.llmApply.checked`.

Для проверки в установленном Google Chrome: установите `playwright` в окружение
разработчика (`uv pip install --python .venv/Scripts/python.exe playwright`),
запустите сайт и выполните:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/check_overlay_browser.py --run-id ID
# Дополнительно отправить реальное изображение в Qwen через кнопку сайта:
.venv/Scripts/python.exe scripts/check_overlay_browser.py --run-id ID --vlm
```

Замените `ID` идентификатором завершённого запуска с непустой разметкой.
Тест проверяет пиксели canvas, скрытие слоёв, resize и повторную загрузку изображения.
Скриншоты сохраняются в `temp/overlay-browser.png` и `temp/qwen-browser.png`.

### Manuscript Mask2Former Lines (ветка v_0_1_13)

`bat/INSTALL_MANUSCRIPT.bat` устанавливает библиотеку из локального checkout в `.venv-manuscript` и проверяет её ONNX-модель на GPU. В интерфейсе — **Manuscript Mask2Former Lines · ONNX**, в API — `manuscript_mask2former_onnx`. [Веса, исправление памяти и результаты проверок](docs/manuscript-mask2former.md).
