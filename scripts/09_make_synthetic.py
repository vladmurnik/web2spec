"""Generate a deterministic, offline synthetic HTML -> JSON extraction dataset.

Pages mimic real product, article, listing, job and real-estate pages: semantic
markup, navigation/header/footer, JSON-LD and scripts as noise. Every target
value is present in the visible part of the compressed HTML.

Usage:
    python scripts/09_make_synthetic.py --out data/raw/synthetic --count 500
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from web2spec.html_compress import compress_html  # noqa: E402
from web2spec.schema import Sample, length_stats, make_id  # noqa: E402
from web2spec.tokens import load_tokenizer, token_lengths  # noqa: E402

SEED = 42
MAX_CHARS = 3500
MIN_CHARS = 1500
CATEGORIES = ("product", "article", "listing", "job", "realestate")
LANGS = ("ru", "en")
PIPELINE_VERSION = "synthetic-0.1.0"
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")

INSTRUCTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "product": {
        "ru": (
            "Извлеки из HTML карточки товара название, цену, валюту, наличие и бренд в JSON.",
            "Преобразуй страницу товара в JSON с полями title, price, currency, availability.",
            "Собери структурированные данные о товаре. Верни только JSON без пояснений.",
            "Прочитай HTML и верни JSON: название, цена, валюта, рейтинг и число отзывов.",
            "Какие данные о товаре есть на странице? Ответь JSON с title, price и currency.",
        ),
        "en": (
            "Extract the product title, price, currency, availability and brand as JSON.",
            "Convert the product page to JSON with title, price, currency and availability.",
            "Return structured product data as JSON only, without explanations.",
            "Read the HTML and return JSON with title, price, currency, rating and review count.",
            "What product data is on the page? Answer with JSON: title, price, currency.",
        ),
    },
    "article": {
        "ru": (
            "Извлеки из HTML статьи заголовок, автора, дату публикации и раздел в JSON.",
            "Собери метаданные статьи: title, author, published_at, section, description.",
            "Преобразуй страницу статьи в JSON с заголовком, автором и датой публикации.",
            "Прочитай HTML и верни JSON: заголовок, автор, даты, рубрика и краткое описание.",
            "Какие метаданные есть у статьи? Ответь JSON по схеме.",
        ),
        "en": (
            "Extract the article headline, author, publication date and section as JSON.",
            "Collect article metadata: title, author, published_at, section and description.",
            "Convert the article page to JSON with headline, author and publication date.",
            "Read the HTML and return JSON with title, author, dates, section and summary.",
            "What metadata does the article have? Answer with JSON following the schema.",
        ),
    },
    "listing": {
        "ru": (
            "Извлеки из HTML список объявлений: количество, заголовки, ссылки и цены в JSON.",
            "Собери элементы листинга в JSON: count и items с title, url, price, currency.",
            "Преобразуй страницу со списком предложений в JSON с массивом items.",
            "Прочитай HTML и верни JSON со всеми объявлениями: название, ссылка и цена.",
            "Сколько объявлений на странице? Верни JSON с count и items.",
        ),
        "en": (
            "Extract the listings: count, titles, links and prices as JSON.",
            "Build listing items as JSON: count and items with title, url, price and currency.",
            "Convert the offers page to JSON with an items array.",
            "Read the HTML and return JSON with every listing: title, link and price.",
            "How many listings are on the page? Answer with JSON containing count and items.",
        ),
    },
    "job": {
        "ru": (
            "Извлеки из HTML вакансии название, компанию, город, зарплату и дату в JSON.",
            "Собери данные вакансии: title, company, location, salary, currency, posted_at.",
            "Преобразуй страницу вакансии в JSON с зарплатой и типом занятости.",
            "Прочитай HTML и верни JSON: должность, компания, локация, зарплата и дата.",
            "Какие условия указаны в вакансии? Ответь JSON по схеме.",
        ),
        "en": (
            "Extract the job title, company, city, salary and posting date as JSON.",
            "Collect job data: title, company, location, salary, currency and posted_at.",
            "Convert the vacancy page to JSON with salary and employment type.",
            "Read the HTML and return JSON with role, company, location, salary and date.",
            "What terms are listed in the vacancy? Answer with JSON following the schema.",
        ),
    },
    "realestate": {
        "ru": (
            "Извлеки из HTML объявления заголовок, цену, адрес, площадь и комнаты в JSON.",
            "Собери параметры объекта: title, price, currency, address, area, rooms.",
            "Преобразуй страницу объекта в JSON с адресом, площадью и числом комнат.",
            "Прочитай HTML и верни JSON: заголовок, цена, адрес, площадь и комнаты.",
            "Какие параметры недвижимости указаны? Ответь JSON по схеме.",
        ),
        "en": (
            "Extract the property title, price, address, area and room count as JSON.",
            "Collect property details: title, price, currency, address, area and rooms.",
            "Convert the property page to JSON with address, floor area and room count.",
            "Read the HTML and return JSON with title, price, address, area and rooms.",
            "What property details are listed? Answer with JSON following the schema.",
        ),
    },
}

SITES = {
    ("product", "ru"): "ТехноМаркет",
    ("product", "en"): "Northline Store",
    ("article", "ru"): "Городские новости",
    ("article", "en"): "The Daily Ledger",
    ("listing", "ru"): "Доска объявлений",
    ("listing", "en"): "Marketplace",
    ("job", "ru"): "Работа рядом",
    ("job", "en"): "Career Hub",
    ("realestate", "ru"): "Дом и Квартира",
    ("realestate", "en"): "Urban Nest",
}

PROMOS = {
    ("product", "ru"): "Скидки до 40% · доставка от 990 ₽",
    ("product", "en"): "Up to 40% off · delivery from $9",
    ("article", "ru"): "Городская повестка · архив с 2020 года",
    ("article", "en"): "Local affairs · archive since 2020",
    ("listing", "ru"): "Свежие объявления от 500 ₽",
    ("listing", "en"): "Fresh listings from £5",
    ("job", "ru"): "Новые вакансии с зарплатой от 45 000 ₽",
    ("job", "en"): "New roles paying from $2,400",
    ("realestate", "ru"): "Объекты от 4 900 000 ₽",
    ("realestate", "en"): "Homes from £210,000",
}

NAV = {
    "ru": ("Главная", "Каталог", "Доставка", "Контакты"),
    "en": ("Home", "Catalog", "Delivery", "Contacts"),
}
FOOTER_LINKS = {
    "ru": ("О компании", "Помощь", "Возврат", "Вакансии"),
    "en": ("About", "Help", "Returns", "Careers"),
}

PRODUCT_KINDS = (
    {
        "ru": "Беспроводные наушники",
        "en": "Wireless headphones",
        "base_ru": 6990,
        "base_en": 89,
        "sku": "AUD",
        "specs": (
            ("Тип", "Type", "накладные", "over-ear"),
            ("Подключение", "Connectivity", "Bluetooth 5.3", "Bluetooth 5.3"),
            ("Время работы", "Battery life", "до 32 часов", "up to 32 hours"),
            ("Вес", "Weight", "248 г", "248 g"),
        ),
    },
    {
        "ru": "Электрический чайник",
        "en": "Electric kettle",
        "base_ru": 3490,
        "base_en": 49,
        "sku": "KIT",
        "specs": (
            ("Объём", "Capacity", "1,7 л", "1.7 L"),
            ("Мощность", "Power", "2200 Вт", "2200 W"),
            ("Материал", "Material", "нержавеющая сталь", "stainless steel"),
            ("Гарантия", "Warranty", "2 года", "2 years"),
        ),
    },
    {
        "ru": "Городской рюкзак",
        "en": "Urban backpack",
        "base_ru": 4990,
        "base_en": 69,
        "sku": "BAG",
        "specs": (
            ("Объём", "Capacity", "24 л", "24 L"),
            ("Материал", "Material", "водоотталкивающий нейлон", "water-resistant nylon"),
            ("Отделение для ноутбука", "Laptop sleeve", "15,6 дюйма", "15.6 inch"),
            ("Вес", "Weight", "780 г", "780 g"),
        ),
    },
    {
        "ru": "Монитор 27 дюймов",
        "en": "27-inch monitor",
        "base_ru": 24990,
        "base_en": 299,
        "sku": "MON",
        "specs": (
            ("Диагональ", "Diagonal", "27 дюймов", "27 inch"),
            ("Разрешение", "Resolution", "2560x1440", "2560x1440"),
            ("Частота", "Refresh rate", "75 Гц", "75 Hz"),
            ("Матрица", "Panel", "IPS", "IPS"),
        ),
    },
    {
        "ru": "Робот-пылесос",
        "en": "Robot vacuum",
        "base_ru": 32990,
        "base_en": 399,
        "sku": "VAC",
        "specs": (
            ("Мощность всасывания", "Suction power", "2600 Па", "2600 Pa"),
            ("Время работы", "Runtime", "90 минут", "90 minutes"),
            ("Пылесборник", "Dust bin", "0,5 л", "0.5 L"),
            ("Управление", "Control", "приложение", "mobile app"),
        ),
    },
    {
        "ru": "Механическая клавиатура",
        "en": "Mechanical keyboard",
        "base_ru": 7990,
        "base_en": 109,
        "sku": "KEY",
        "specs": (
            ("Раскладка", "Layout", "75 клавиш", "75 keys"),
            ("Переключатели", "Switches", "brown", "brown"),
            ("Подсветка", "Backlight", "RGB", "RGB"),
            ("Подключение", "Connection", "USB-C", "USB-C"),
        ),
    },
    {
        "ru": "Кофемашина рожковая",
        "en": "Espresso machine",
        "base_ru": 28990,
        "base_en": 349,
        "sku": "COF",
        "specs": (
            ("Давление", "Pressure", "15 бар", "15 bar"),
            ("Объём резервуара", "Water tank", "1,8 л", "1.8 L"),
            ("Мощность", "Power", "1450 Вт", "1450 W"),
            ("Капучинатор", "Milk frother", "автоматический", "automatic"),
        ),
    },
    {
        "ru": "Кроссовки для бега",
        "en": "Running shoes",
        "base_ru": 8990,
        "base_en": 119,
        "sku": "RUN",
        "specs": (
            ("Размер", "Size", "42", "9 UK"),
            ("Материал", "Material", "сетка и текстиль", "mesh and textile"),
            ("Подошва", "Sole", "амортизирующая пена", "cushioning foam"),
            ("Вес", "Weight", "265 г", "265 g"),
        ),
    },
    {
        "ru": "Настольная лампа",
        "en": "Desk lamp",
        "base_ru": 2490,
        "base_en": 39,
        "sku": "LMP",
        "specs": (
            ("Мощность", "Power", "10 Вт", "10 W"),
            ("Цветовая температура", "Colour temperature", "3000-6000 K", "3000-6000 K"),
            ("Питание", "Power supply", "USB-C", "USB-C"),
            ("Материал", "Material", "алюминий", "aluminium"),
        ),
    },
    {
        "ru": "Планетарный миксер",
        "en": "Stand mixer",
        "base_ru": 19990,
        "base_en": 229,
        "sku": "MIX",
        "specs": (
            ("Мощность", "Power", "1000 Вт", "1000 W"),
            ("Объём чаши", "Bowl capacity", "5 л", "5 L"),
            ("Скорости", "Speeds", "8", "8"),
            ("Насадки", "Attachments", "3", "3"),
        ),
    },
)

PRODUCT_MODELS = (
    "Lite",
    "Air",
    "Plus",
    "Pro",
    "Max",
    "Studio",
    "Mini",
    "Eco",
    "Prime",
    "Ultra",
)
PRODUCT_BRANDS = ("Aurora", "Nordwind", "Lumen", "Kraft", "Vega", "Orbis", "Terra", "Kite")

PRODUCT_REVIEWS = {
    "ru": (
        (
            "Анна К.",
            "Пользуюсь месяц, всё работает как заявлено. Упаковка целая, доставка быстрая.",
        ),
        ("Игорь С.", "Хорошее соотношение цены и качества. Есть мелкие недочёты, но я доволен."),
        (
            "Марина В.",
            "Брала в подарок, получатель доволен. Инструкция понятная, гарантия оформлена.",
        ),
    ),
    "en": (
        ("Anna K.", "A month in and everything works as advertised. Packaging was intact."),
        ("Igor S.", "Good value for the price. A few small flaws, but overall I am happy."),
        ("Marina V.", "Bought it as a gift and the recipient loves it. Clear manual and warranty."),
    ),
}

AVAILABILITY = {
    "in_stock": ("В наличии", "In stock"),
    "out_of_stock": ("Нет в наличии", "Out of stock"),
    "preorder": ("Предзаказ", "Pre-order"),
}

ARTICLE_TOPICS = (
    ("Городские власти обновили правила парковки", "City authorities update parking rules"),
    ("В центре открылся новый книжный магазин", "New bookshop opens in the city centre"),
    ("Метро продлило работу в выходные", "Metro extends weekend hours"),
    ("Школы переходят на обновлённую программу", "Schools adopt an updated curriculum"),
    ("Город выделил гранты на ремонт фасадов", "City announces grants for facade repairs"),
    ("Открылся сезон фермерских рынков", "Farmers market season opens"),
    ("Новый парк появится на месте промзоны", "New park planned on former industrial site"),
    ("Больницы обновили оборудование", "Hospitals upgrade medical equipment"),
    ("Стоимость проезда останется прежней", "Public transport fares to stay unchanged"),
    ("Волонтёры очистили берег реки", "Volunteers clean up the riverbank"),
    ("Библиотеки запустили вечерние курсы", "Libraries launch evening classes"),
    ("Ремонт моста завершат досрочно", "Bridge repairs to finish ahead of schedule"),
    ("Городской фестиваль соберёт музыкантов", "City festival to bring together musicians"),
    ("Открылась выставка современного искусства", "Contemporary art exhibition opens"),
    ("Департамент запустил онлайн-приём обращений", "Department launches online requests portal"),
    ("Новые правила выгула собак в парках", "New dog-walking rules in parks"),
    ("Рынок аренды замедлился к осени", "Rental market slows in autumn"),
    ("Город закупит электробусы", "City to buy electric buses"),
    ("Школьники представили проект умного двора", "Students present smart courtyard project"),
    ("Открылся центр поддержки предпринимателей", "Small business support centre opens"),
)

DISTRICTS = {
    "ru": ("Хамовники", "Арбат", "Сокол", "Замоскворечье", "Черёмушки"),
    "en": ("Camden", "Shoreditch", "Greenwich", "Fulham", "Islington"),
}
AUTHORS = {
    "ru": (
        "Иван Петров",
        "Мария Соколова",
        "Алексей Кузнецов",
        "Ольга Ветрова",
        "Дмитрий Лавров",
        "Елена Морозова",
        "Павел Крылов",
        "Наталья Ильина",
        "Сергей Орлов",
        "Анна Белова",
    ),
    "en": (
        "James Whitfield",
        "Clara Bennett",
        "Daniel Hughes",
        "Sophie Turner",
        "Mark Ellison",
        "Ruth Calloway",
        "Peter Nash",
        "Laura Finch",
        "Thomas Reed",
        "Emma Lawson",
    ),
}
SECTIONS = {
    "ru": ("Город", "Экономика", "Общество", "Культура", "Технологии", "Наука"),
    "en": ("City", "Business", "Society", "Culture", "Technology", "Science"),
}

RU_MONTHS = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
EN_MONTHS = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DATES = (
    "2026-09-08",
    "2026-09-15",
    "2026-09-24",
    "2026-10-01",
    "2026-10-06",
    "2026-10-12",
    "2026-10-19",
    "2026-10-23",
    "2026-10-27",
    "2026-10-30",
)


def human_date(iso: str, lang: str) -> str:
    day = date.fromisoformat(iso)
    if lang == "ru":
        return f"{day.day} {RU_MONTHS[day.month]} {day.year}"
    return f"{EN_MONTHS[day.month]} {day.day}, {day.year}"


ARTICLE_BODY = {
    "ru": (
        "По словам представителей администрации, изменения затронут в первую очередь "
        "район {district}. Пилотный проект рассчитан на шесть месяцев, после чего "
        "власти обещают собрать обратную связь жителей.",
        "Эксперты отмечают, что подобные решения обычно влияют на повседневные "
        "маршруты горожан. По предварительным оценкам, спрос на новые услуги может "
        "вырасти на {pct}% уже к концу квартала.",
        "Местные предприниматели отнеслись к инициативе по-разному. Одни рассчитывают "
        "на приток посетителей, другие опасаются дополнительных расходов и просят "
        "предусмотреть переходный период.",
        "В течение месяца пройдут открытые встречи с жителями. Задать вопросы можно "
        "будет онлайн, а также в районном центре услуг; расписание опубликуют на "
        "официальном сайте.",
        "Наблюдатели напоминают, что окончательные параметры проекта ещё могут "
        "измениться. Итоговое решение обещают принять после общественных обсуждений "
        "и независимой оценки.",
    ),
    "en": (
        "Officials said the changes will affect {district} first. The pilot will run for "
        "six months, after which the city promises to collect feedback from residents.",
        "Experts note that similar decisions often change daily routines. Early estimates "
        "suggest demand for the new services could grow by {pct}% by the end of the quarter.",
        "Local business owners are divided. Some expect more visitors, while others worry "
        "about extra costs and ask for a transition period.",
        "Open meetings with residents will be held throughout the month. Questions can be "
        "submitted online or at the district service centre; the schedule is on the website.",
        "Observers point out that the final parameters may still change. A decision is "
        "expected after public consultations and an independent review.",
    ),
}

ARTICLE_QUOTES = {
    "ru": (
        "Главное — не скорость, а предсказуемость: люди должны понимать, что изменится.",
        "Мы будем смотреть на данные каждый месяц и корректировать решение.",
        "Для района это шанс обновить инфраструктуру без резких изменений.",
    ),
    "en": (
        "The priority is predictability: residents should know exactly what changes.",
        "We will review the data every month and adjust the decision if needed.",
        "For the district this is a chance to upgrade infrastructure without shocks.",
    ),
}

ARTICLE_EXPERTS = {
    "ru": ("Антон Лебедев", "Ирина Громова", "Виктор Соловьёв"),
    "en": ("Anton Lebedev", "Irina Gromova", "Victor Solovyov"),
}

LISTING_THEMES = (
    {
        "ru": "Мебель",
        "en": "Furniture",
        "base_ru": 15000,
        "base_en": 190,
        "items": (
            ("Диван трёхместный", "Three-seat sofa"),
            ("Кресло кожаное", "Leather armchair"),
            ("Стол обеденный дубовый", "Oak dining table"),
            ("Шкаф-купе", "Sliding wardrobe"),
            ("Стеллаж металлический", "Metal shelving unit"),
            ("Комод с зеркалом", "Dresser with mirror"),
            ("Кровать двуспальная", "Double bed"),
            ("Журнальный столик", "Coffee table"),
        ),
    },
    {
        "ru": "Электроника",
        "en": "Electronics",
        "base_ru": 25000,
        "base_en": 320,
        "items": (
            ("Ноутбук 14 дюймов", "14-inch laptop"),
            ("Смартфон 128 ГБ", "128 GB smartphone"),
            ("Монитор 24 дюйма", "24-inch monitor"),
            ("Планшет 10 дюймов", "10-inch tablet"),
            ("Фотоаппарат зеркальный", "DSLR camera"),
            ("Колонка портативная", "Portable speaker"),
            ("Принтер лазерный", "Laser printer"),
            ("Наушники накладные", "Over-ear headphones"),
        ),
    },
    {
        "ru": "Велосипеды",
        "en": "Bikes",
        "base_ru": 12000,
        "base_en": 150,
        "items": (
            ("Велосипед горный", "Mountain bike"),
            ("Велосипед городской", "City bike"),
            ("Самокат взрослый", "Adult scooter"),
            ("Самокат детский", "Kids scooter"),
            ("Шлем велосипедный", "Bike helmet"),
            ("Замок для велосипеда", "Bike lock"),
            ("Кресло детское на велосипед", "Child bike seat"),
            ("Насос для шин", "Tyre pump"),
        ),
    },
    {
        "ru": "Для дома",
        "en": "Home",
        "base_ru": 8000,
        "base_en": 100,
        "items": (
            ("Пылесос вертикальный", "Upright vacuum"),
            ("Кофемашина рожковая", "Espresso machine"),
            ("Микроволновая печь", "Microwave oven"),
            ("Блендер стационарный", "Blender"),
            ("Утюг паровой", "Steam iron"),
            ("Обогреватель масляный", "Oil radiator"),
            ("Вентилятор напольный", "Floor fan"),
            ("Швейная машина", "Sewing machine"),
        ),
    },
    {
        "ru": "Детям",
        "en": "Kids",
        "base_ru": 6000,
        "base_en": 75,
        "items": (
            ("Коляска прогулочная", "Pushchair"),
            ("Кроватка детская", "Cot"),
            ("Автокресло детское", "Child car seat"),
            ("Конструктор большой", "Large building set"),
            ("Самокат трёхколёсный", "Three-wheel scooter"),
            ("Рюкзак школьный", "School backpack"),
            ("Стол детский", "Kids desk"),
            ("Манеж складной", "Foldable playpen"),
        ),
    },
    {
        "ru": "Инструменты",
        "en": "Tools",
        "base_ru": 9000,
        "base_en": 110,
        "items": (
            ("Перфоратор", "Rotary hammer"),
            ("Дрель аккумуляторная", "Cordless drill"),
            ("Набор отвёрток", "Screwdriver set"),
            ("Пила циркулярная", "Circular saw"),
            ("Шуруповёрт", "Impact driver"),
            ("Строительный пылесос", "Construction vacuum"),
            ("Уровень лазерный", "Laser level"),
            ("Компрессор автомобильный", "Car compressor"),
        ),
    },
    {
        "ru": "Одежда",
        "en": "Fashion",
        "base_ru": 4000,
        "base_en": 50,
        "items": (
            ("Куртка зимняя", "Winter jacket"),
            ("Пальто шерстяное", "Wool coat"),
            ("Кроссовки беговые", "Running shoes"),
            ("Ботинки кожаные", "Leather boots"),
            ("Сумка кожаная", "Leather bag"),
            ("Джинсы прямые", "Straight jeans"),
            ("Свитер вязаный", "Knitted sweater"),
            ("Платье вечернее", "Evening dress"),
        ),
    },
    {
        "ru": "Музыка",
        "en": "Music",
        "base_ru": 15000,
        "base_en": 190,
        "items": (
            ("Гитара акустическая", "Acoustic guitar"),
            ("Синтезатор 61 клавиша", "61-key synthesizer"),
            ("Укулеле", "Ukulele"),
            ("Микрофон студийный", "Studio microphone"),
            ("Аудиоинтерфейс USB", "USB audio interface"),
            ("Стойка для микрофона", "Microphone stand"),
            ("Педаль эффектов", "Effects pedal"),
            ("Наушники студийные", "Studio headphones"),
        ),
    },
    {
        "ru": "Книги",
        "en": "Books",
        "base_ru": 800,
        "base_en": 12,
        "items": (
            ("Собрание сочинений", "Collected works"),
            ("Учебник английского", "English textbook"),
            ("Энциклопедия искусства", "Art encyclopedia"),
            ("Комиксы, комплект", "Comic book set"),
            ("Словарь толковый", "Dictionary"),
            ("Книги детские", "Children's books"),
            ("Роман современный", "Contemporary novel"),
            ("Путеводитель", "Travel guide"),
        ),
    },
    {
        "ru": "Сад",
        "en": "Garden",
        "base_ru": 7000,
        "base_en": 90,
        "items": (
            ("Газонокосилка", "Lawn mower"),
            ("Триммер садовый", "Garden trimmer"),
            ("Мангал складной", "Folding grill"),
            ("Качели садовые", "Garden swing"),
            ("Набор садовых инструментов", "Garden tool set"),
            ("Куст роз", "Rose bush"),
            ("Теплица каркасная", "Framed greenhouse"),
            ("Садовый шланг", "Garden hose"),
        ),
    },
)

CONDITIONS = {
    "ru": ("отличное состояние", "хорошее состояние", "как новое", "без следов использования"),
    "en": ("excellent condition", "good condition", "like new", "no signs of use"),
}
CITIES = {
    "ru": ("Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск", "Самара"),
    "en": ("London", "Manchester", "Bristol", "Leeds", "Edinburgh", "Glasgow"),
}

JOB_ROLES = {
    "ru": (
        ("Frontend-разработчик", 180000),
        ("Backend-разработчик", 220000),
        ("Аналитик данных", 160000),
        ("Менеджер по продажам", 120000),
        ("Бухгалтер", 90000),
        ("Дизайнер интерфейсов", 150000),
        ("Инженер-конструктор", 130000),
        ("Маркетолог", 110000),
        ("HR-специалист", 85000),
        ("Курьер", 65000),
        ("Водитель категории B", 80000),
        ("Повар", 75000),
        ("Бариста", 55000),
        ("Администратор офиса", 70000),
        ("Тестировщик", 140000),
        ("Системный администратор", 100000),
        ("Логист", 95000),
        ("Экономист", 105000),
        ("Юрист", 125000),
        ("Учитель математики", 60000),
    ),
    "en": (
        ("Frontend developer", 6500),
        ("Backend developer", 7200),
        ("Data analyst", 5800),
        ("Sales manager", 5000),
        ("Accountant", 4200),
        ("Product designer", 6000),
        ("Mechanical engineer", 5500),
        ("Marketing manager", 4800),
        ("HR specialist", 4000),
        ("Courier", 2600),
        ("Delivery driver", 3200),
        ("Chef", 3000),
        ("Barista", 2400),
        ("Office administrator", 3100),
        ("QA engineer", 5600),
        ("System administrator", 4500),
        ("Logistics coordinator", 3800),
        ("Economist", 4700),
        ("Lawyer", 6100),
        ("Maths teacher", 3300),
    ),
}
COMPANIES = {
    "ru": (
        "ООО «Северный ветер»",
        "АО «Городские сети»",
        "ООО «ТехноПарк»",
        "ООО «Логика»",
        "АО «Восход»",
        "ООО «Зелёный двор»",
        "ООО «Медтехника»",
        "ООО «Профит»",
        "АО «Мостстрой»",
        "ООО «Книжный дом»",
    ),
    "en": (
        "Northwind Ltd",
        "City Networks",
        "TechnoPark",
        "Logic Works",
        "Sunrise Group",
        "Green Yard",
        "MedTech Solutions",
        "Profit Partners",
        "BridgeBuild",
        "Book House",
    ),
}
JOB_CITIES = {
    "ru": ("Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск", "Самара"),
    "en": ("London", "Manchester", "Bristol", "Leeds", "Edinburgh", "Glasgow"),
}
EMPLOYMENT = (
    ("FULL_TIME", "Полная занятость", "Full-time"),
    ("PART_TIME", "Частичная занятость", "Part-time"),
    ("CONTRACT", "Проектная работа", "Contract"),
    ("INTERNSHIP", "Стажировка", "Internship"),
)

REALESTATE_DISTRICTS = {
    "ru": ("Хамовники", "Арбат", "Сокол", "Замоскворечье", "Черёмушки"),
    "en": ("Camden", "Shoreditch", "Greenwich", "Fulham", "Islington"),
}
REALESTATE_STREETS = {
    "ru": ("Льва Толстого", "Арбат", "Ленинградский проспект", "Пятницкая", "Профсоюзная"),
    "en": ("Ashcroft Road", "Brick Lane", "Creekside", "Fulham Road", "Highbury Grove"),
}
PRICE_PER_SQM = {
    "ru": (420000, 510000, 360000, 390000, 300000),
    "en": (9500, 8200, 7400, 8800, 7900),
}

PRODUCT_FILLER = {
    "ru": (
        "Доставка по городу занимает один-два дня, самовывоз доступен ежедневно "
        "с 10:00 до 21:00. При заказе от 5 000 ₽ доставка бесплатная.",
        "Если товар не подошёл, его можно вернуть в течение 14 дней при сохранении "
        "упаковки и комплектации. Возврат оформляется в личном кабинете.",
        "Оплатить заказ можно картой, переводом или при получении. Для юридических "
        "лиц доступен счёт и закрывающие документы.",
    ),
    "en": (
        "City delivery takes one to two days, and pickup is available daily from 10:00 "
        "to 21:00. Orders over $50 ship free.",
        "If the item does not fit, it can be returned within 14 days with the original "
        "packaging and accessories. Returns are handled in your account.",
        "Pay by card, bank transfer or on delivery. Business customers can request an "
        "invoice and full accounting documents.",
    ),
}
LISTING_FILLER = {
    "ru": (
        "Сделка проходит через безопасный сервис: деньги перечисляются продавцу "
        "после подтверждения получения товара.",
        "Перед покупкой проверьте товар при встрече. Не переводите предоплату "
        "незнакомым продавцам и сохраняйте переписку.",
        "Объявления в этом разделе обновляются ежедневно. Сохраните поиск, чтобы "
        "получать уведомления о новых предложениях.",
    ),
    "en": (
        "The deal can run through the secure service: funds are released to the seller "
        "after the buyer confirms delivery.",
        "Inspect the item in person before paying. Never send a deposit to an unknown "
        "seller and keep the conversation history.",
        "Listings in this section are updated daily. Save your search to get alerts "
        "about new offers.",
    ),
}
JOB_FILLER = {
    "ru": (
        "Компания предлагает официальное трудоустройство, оплачиваемый отпуск и "
        "обучение за счёт работодателя. Гибридный формат обсуждается на собеседовании.",
        "Откликнуться можно на сайте или отправить резюме на почту. Мы связываемся "
        "со всеми кандидатами в течение пяти рабочих дней.",
        "На собеседовании расскажем о команде, задачах и этапах отбора. Тестовое "
        "задание оплачивается и занимает не больше двух часов.",
    ),
    "en": (
        "The company offers a permanent contract, paid holiday and employer-funded "
        "training. Hybrid working can be discussed at the interview.",
        "Apply on the website or email your CV. We reply to every candidate within "
        "five working days.",
        "The interview covers the team, the roadmap and the hiring stages. The take-home "
        "task is paid and takes no more than two hours.",
    ),
}
REALESTATE_FILLER = {
    "ru": (
        "Показы проводятся по предварительной записи, в том числе вечером и в выходные. "
        "Документы готовы к сделке, обременений нет.",
        "Рядом находятся школа, поликлиника и парк. До метро можно дойти за 12 минут, "
        "до центра — около 20 минут на автомобиле.",
        "Ипотеку можно оформить через партнёрские банки. Поможем с оценкой и "
        "юридическим сопровождением сделки.",
    ),
    "en": (
        "Viewings are by appointment, including evenings and weekends. The paperwork "
        "is ready and the property has no outstanding charges.",
        "A school, a medical centre and a park are nearby. The nearest station is a "
        "12-minute walk and the centre is about 20 minutes by car.",
        "Mortgages are available through partner lenders. We can help with valuation "
        "and legal support throughout the purchase.",
    ),
}

def format_price(amount: int | float, lang: str, currency: str) -> str:
    number = int(amount)
    if lang == "ru" and currency == "RUB":
        return f"{number:,}".replace(",", " ") + " ₽"
    symbol = {"USD": "$", "EUR": "€", "GBP": "£", "RUB": "₽"}.get(currency, "")
    return f"{symbol}{number:,}"


def _complexity(rng: random.Random) -> int:
    return rng.choices((1, 2, 3), weights=(3, 5, 2), k=1)[0]


def _chrome_header(lang: str, site: str, promo: str) -> str:
    labels = NAV[lang]
    if lang == "ru":
        phone = "+7 495 123-45-67"
        search = "Поиск по сайту"
        aria = "Основная навигация"
    else:
        phone = "+44 20 7946 0123"
        search = "Search the site"
        aria = "Main navigation"
    items = "".join(f'<li><a href="/{i + 1}">{label}</a></li>' for i, label in enumerate(labels))
    return (
        '<header class="site-header">'
        f'<div class="topbar"><a class="logo" href="/">{site}</a>'
        f'<span class="tagline">{promo}</span>'
        f'<a class="phone" href="tel:000">{phone}</a></div>'
        f'<nav class="main-nav" aria-label="{aria}"><ul>{items}</ul></nav>'
        f'<form class="search" role="search"><input name="q" placeholder="{search}">'
        f"<button type=\"submit\">{labels[1]}</button></form>"
        "</header>"
    )


def _chrome_footer(lang: str, site: str) -> str:
    links = "".join(f'<li><a href="/info/{i + 1}">{label}</a></li>' for i, label in
                    enumerate(FOOTER_LINKS[lang]))
    if lang == "ru":
        address = "Москва, ул. Примерная, 1"
        legal = f"© 2026 {site}. Все права защищены. Информация не является офертой."
        heading = "Сервис"
    else:
        address = "1 Example Street, London"
        legal = f"© 2026 {site}. All rights reserved. Information is not an offer."
        heading = "Service"
    return (
        '<footer class="site-footer">'
        f'<div class="footer-col"><h3>{heading}</h3><ul>{links}</ul></div>'
        f'<div class="footer-col"><h3>{"Контакты" if lang == "ru" else "Contacts"}</h3>'
        f"<p>{address}</p><p>+7 495 123-45-67</p></div>"
        f'<p class="legal">{legal}</p>'
        "</footer>"
    )


@dataclass(slots=True)
class Candidate:
    category: str
    lang: str
    idx: int
    title: str
    target: dict[str, Any]
    core: str
    extras: list[str]
    filler: list[str]
    complexity: int
    site: str
    slug: str
    jsonld: dict[str, Any]
    instruction: str


def _inner_nav(lang: str, site: str) -> str:
    labels = NAV[lang]
    items = "".join(
        f'<li><a href="/c/{i + 1}">{label}</a></li>' for i, label in enumerate(labels)
    )
    aria = "Основная навигация" if lang == "ru" else "Main navigation"
    return f'<nav class="crumbs" aria-label="{aria}"><ul>{items}</ul></nav>'


def _inner_header(lang: str, site: str, promo: str) -> str:
    section = "Каталог" if lang == "ru" else "Catalog"
    return (
        '<header class="content-head">'
        f'<span class="section">{section}</span>'
        f'<span class="promo-line">{site}: {promo}</span>'
        "</header>"
    )


def _inner_footer(lang: str, site: str) -> str:
    if lang == "ru":
        legal = f"© 2026 {site}. Информация не является офертой."
        note = "Подписка на уведомления — от 199 ₽ в месяц, цены справочные."
        heading = "Информация"
    else:
        legal = f"© 2026 {site}. Information is not an offer."
        note = "Alerts subscription — from $3 per month; prices are indicative only."
        heading = "Information"
    links = "".join(
        f'<li><a href="/info/{i + 1}">{label}</a></li>'
        for i, label in enumerate(FOOTER_LINKS[lang])
    )
    return (
        '<footer class="content-foot">'
        f"<h3>{heading}</h3><ul>{links}</ul>"
        f"<p>{note}</p><p class=\"legal\">{legal}</p>"
        "</footer>"
    )


def render_document(cand: Candidate, extras: list[str], filler_count: int) -> str:
    site = cand.site
    head_title = f"{cand.title} — {site}"
    payload = json.dumps(cand.jsonld, ensure_ascii=False)
    state = json.dumps(
        {"cart": 0, "recommendedPrice": 1290, "abVariant": "b", "userId": None},
        ensure_ascii=False,
    )
    body = (
        _chrome_header(cand.lang, site, PROMOS[(cand.category, cand.lang)])
        + '<main class="page-main" role="main">'
        + _inner_header(cand.lang, site, PROMOS[(cand.category, cand.lang)])
        + _inner_nav(cand.lang, site)
        + cand.core
        + "".join(extras)
        + "".join(cand.filler[:filler_count])
        + _inner_footer(cand.lang, site)
        + "</main>"
        + _chrome_footer(cand.lang, site)
        + '<script src="/static/app.js" defer></script>'
        + "<script>window.__STATE__ = " + state + ";</script>"
    )
    return (
        "<!doctype html>"
        f'<html lang="{cand.lang}"><head><meta charset="utf-8">'
        f"<title>{head_title}</title>"
        f'<meta name="description" content="{cand.title}">'
        '<link rel="stylesheet" href="/static/app.css">'
        f'<script type="application/ld+json">{payload}</script>'
        "</head><body>"
        + body
        + "</body></html>"
    )


def build_page(cand: Candidate) -> tuple[str, str]:
    """Render and compress one candidate, keeping the length inside the contract."""
    extras = list(cand.extras)
    fill = 0
    html = render_document(cand, extras, fill)
    compressed = compress_html(html, max_chars=MAX_CHARS)
    while len(compressed) < MIN_CHARS and fill < len(cand.filler):
        fill += 1
        html = render_document(cand, extras, fill)
        compressed = compress_html(html, max_chars=MAX_CHARS)
    while len(compressed) > MAX_CHARS - 10 and extras:
        extras.pop()
        html = render_document(cand, extras, fill)
        compressed = compress_html(html, max_chars=MAX_CHARS)
    return html, compressed


def product_candidate(lang: str, idx: int, rng: random.Random) -> Candidate:
    kind = PRODUCT_KINDS[(idx - 1) % len(PRODUCT_KINDS)]
    model = PRODUCT_MODELS[((idx - 1) // len(PRODUCT_KINDS)) % len(PRODUCT_MODELS)]
    brand = PRODUCT_BRANDS[(idx * 3) % len(PRODUCT_BRANDS)]
    title = f"{kind[lang]} {brand} {model}"
    complexity = _complexity(rng)
    price = int(kind["base_" + lang] + rng.randint(0, 900) + (idx % 7) * 130)
    old_price = int(price * 1.2)
    currency = "RUB" if lang == "ru" else ("USD", "EUR", "GBP")[idx % 3]
    stock = rng.choices(("in_stock", "out_of_stock", "preorder"), weights=(6, 1, 1), k=1)[0]
    rating = round(4.0 + rng.randint(0, 9) / 10, 1)
    reviews = rng.randint(14, 860)
    sku = f"{kind['sku']}-{1000 + (idx * 37) % 9000}"
    specs = kind["specs"]
    labels = (0, 1) if lang == "ru" else (1, 1)
    spec_rows = "".join(
        f"<dt>{row[labels[0]]}</dt><dd>{row[2] if lang == 'ru' else row[3]}</dd>"
        for row in specs
    )
    availability = AVAILABILITY[stock][0 if lang == "ru" else 1]
    price_text = format_price(price, lang, currency)
    old_text = format_price(old_price, lang, currency)
    rating_line = ""
    if complexity >= 3:
        if lang == "ru":
            rating_line = (
                f'<p class="rating">Рейтинг {rating:.1f} из 5 · {reviews} отзывов</p>'
            )
        else:
            rating_line = f'<p class="rating">Rating {rating:.1f} out of 5 · {reviews} reviews</p>'
    if lang == "ru":
        brand_line = f'<p class="brand">Бренд: <a href="/brand">{brand}</a></p>'
        stock_line = f'<p class="availability" data-testid="{stock}">{availability}</p>'
        delivery = "<p class=\"delivery\">Доставка завтра · самовывоз сегодня</p>"
        desc_head = "Описание"
        specs_head = "Характеристики"
        reviews_head = "Отзывы покупателей"
        related_head = "Похожие товары"
        currency_label = "Валюта"
        sku_label = "Артикул"
        brand_label = "Бренд"
    else:
        brand_line = f'<p class="brand">Brand: <a href="/brand">{brand}</a></p>'
        stock_line = f'<p class="availability" data-testid="{stock}">{availability}</p>'
        delivery = '<p class="delivery">Next-day delivery · pickup today</p>'
        desc_head = "Description"
        specs_head = "Specifications"
        reviews_head = "Customer reviews"
        related_head = "Related products"
        currency_label = "Currency"
        sku_label = "SKU"
        brand_label = "Brand"
    if lang == "ru":
        desc1 = (
            f"{title} — практичная модель бренда {brand} для повседневного использования. "
            "Корпус собран аккуратно, элементы управления понятны без инструкции, а комплект "
            "поставки включает всё необходимое для начала работы."
        )
        desc2 = (
            "Гарантия производителя — 2 года. Товар прошёл проверку качества и доступен "
            "для доставки по всей стране."
        )
    else:
        desc1 = (
            f"{title} by {brand} is a practical everyday model. The body is neatly assembled, "
            "the controls are easy to use without a manual, and the box includes everything "
            "needed to get started."
        )
        desc2 = (
            "The manufacturer offers a two-year warranty. The item has passed quality checks "
            "and is available for nationwide delivery."
        )
    core = (
        '<div class="breadcrumbs"><a href="/">'
        + ("Главная" if lang == "ru" else "Home")
        + ' / <a href="/catalog">'
        + ("Каталог" if lang == "ru" else "Catalog")
        + f"</a> / <span>{title}</span></div>"
        '<div class="product-layout"><div class="product-gallery">'
        f'<img src="/img/{cand_slug(idx)}.jpg" alt="{title}">'
        '<span class="badge">Хит</span></div>'
        '<div class="product-summary">'
        f"{brand_line}<h1>{title}</h1>{rating_line}"
        f'<div class="price-row"><span class="price">{price_text}</span>'
        f'<span class="old-price">{old_text}</span>'
        '<span class="discount">-20%</span></div>'
        f"{stock_line}{delivery}"
        f'<dl class="key-specs"><dt>{sku_label}</dt><dd>{sku}</dd>'
        f"<dt>{brand_label}</dt><dd>{brand}</dd>"
        f"<dt>{currency_label}</dt><dd>{currency}</dd>"
        f"{spec_rows}</dl>"
        '<div class="buy-row"><button type="button">'
        + ("В корзину" if lang == "ru" else "Add to cart")
        + "</button></div></div></div>"
    )
    review_html = "".join(
        f'<div class="review"><p class="review-author">{author}</p><p>{text}</p></div>'
        for author, text in PRODUCT_REVIEWS[lang]
    )
    related = "".join(
        f'<li><a href="/product/{idx + n}">{PRODUCT_KINDS[(idx + n) % 10][lang]}</a></li>'
        for n in (1, 2, 3)
    )
    extras = [
        f'<section class="description"><h2>{desc_head}</h2><p>{desc1}</p><p>{desc2}</p>'
        "</section>",
    ]
    if complexity >= 2:
        extras.append(
            f'<section class="specs"><h2>{specs_head}</h2><table><tbody>'
            + "".join(
                f"<tr><th>{row[0] if lang == 'ru' else row[1]}</th>"
                f"<td>{row[2] if lang == 'ru' else row[3]}</td></tr>"
                for row in specs
            )
            + "</tbody></table></section>"
        )
    if complexity >= 3:
        extras.append(
            f'<section class="reviews"><h2>{reviews_head}</h2>{review_html}</section>'
            f'<section class="related"><h2>{related_head}</h2><ul>{related}</ul></section>'
        )
    target: dict[str, Any] = {"title": title, "price": price, "currency": currency}
    if complexity >= 2:
        target["availability"] = stock
        target["brand"] = brand
        target["sku"] = sku
    if complexity >= 3:
        target["rating"] = rating
        target["review_count"] = reviews
    jsonld = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": title,
        "brand": {"@type": "Brand", "name": brand},
        "sku": sku,
        "offers": {
            "@type": "Offer",
            "price": price,
            "priceCurrency": currency,
            "availability": f"https://schema.org/{stock}",
        },
    }
    if complexity >= 3:
        jsonld["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": rating,
            "reviewCount": reviews,
        }
    return Candidate(
        category="product",
        lang=lang,
        idx=idx,
        title=title,
        target=target,
        core=core,
        extras=extras,
        filler=list(PRODUCT_FILLER[lang]),
        complexity=complexity,
        site=SITES[("product", lang)],
        slug=f"product-{idx:04d}",
        jsonld=jsonld,
        instruction=INSTRUCTIONS["product"][lang][(idx - 1) % 5],
    )


def cand_slug(idx: int) -> str:
    return f"p{idx:04d}"


def article_candidate(lang: str, idx: int, rng: random.Random) -> Candidate:
    topic_ru, topic_en = ARTICLE_TOPICS[(idx - 1) % len(ARTICLE_TOPICS)]
    variant = (idx - 1) // len(ARTICLE_TOPICS)
    district = DISTRICTS[lang][variant % len(DISTRICTS[lang])]
    base_title = topic_ru if lang == "ru" else topic_en
    title = base_title if variant == 0 else f"{base_title} — {district}"
    complexity = _complexity(rng)
    author = AUTHORS[lang][(idx * 7) % len(AUTHORS[lang])]
    section = SECTIONS[lang][(idx + len(base_title)) % len(SECTIONS[lang])]
    published = DATES[(idx * 3) % len(DATES)]
    modified = (date.fromisoformat(published) + timedelta(days=1 + idx % 4)).isoformat()
    pct = 8 + (idx * 3) % 22
    if lang == "ru":
        lede = (
            f"Коротко: {title[0].lower() + title[1:]}. В районе {district} рассказали, "
            "какие изменения вступят в силу в ближайшие недели."
        )
    else:
        lede = (
            f"In brief: {title[0].lower() + title[1:]}. Officials in {district} explained "
            "which changes will take effect in the coming weeks."
        )
    paragraphs = "".join(
        f"<p>{text.format(district=district, pct=pct)}</p>"
        for text in ARTICLE_BODY[lang]
    )
    quote = ARTICLE_QUOTES[lang][idx % len(ARTICLE_QUOTES[lang])]
    expert = ARTICLE_EXPERTS[lang][idx % len(ARTICLE_EXPERTS[lang])]
    tags = "".join(
        f'<li><a href="/tag/{n}">{tag}</a></li>'
        for n, tag in enumerate(
            (section, district, "город" if lang == "ru" else "city"), start=1
        )
    )
    related = "".join(
        f'<li><a href="/news/{idx + n}">'
        f"{ARTICLE_TOPICS[(idx + n) % len(ARTICLE_TOPICS)][0 if lang == 'ru' else 1]}"
        "</a></li>"
        for n in (1, 2, 3)
    )
    if lang == "ru":
        kicker = f'<p class="kicker"><a href="/section">{section}</a></p>'
        byline = f'<p class="byline">Автор: <a rel="author">{author}</a></p>'
        dates = (
            f'<p class="dates"><time datetime="{published}">{human_date(published, lang)}</time>'
            f" · Опубликовано: {published}</p>"
        )
        updated = (
            f'<p class="updated">Обновлено: {modified}</p>' if complexity >= 3 else ""
        )
        tags_head = "Теги"
        related_head = "Читайте также"
        quote_role = "эксперт по городскому развитию"
    else:
        kicker = f'<p class="kicker"><a href="/section">{section}</a></p>'
        byline = f'<p class="byline">By <a rel="author">{author}</a></p>'
        dates = (
            f'<p class="dates"><time datetime="{published}">{human_date(published, lang)}</time>'
            f" · Published: {published}</p>"
        )
        updated = f'<p class="updated">Updated: {modified}</p>' if complexity >= 3 else ""
        tags_head = "Tags"
        related_head = "Read next"
        quote_role = "urban policy expert"
    core = (
        '<article class="news-article"><header class="article-head">'
        f"{kicker}<h1>{title}</h1>{byline}{dates}{updated}"
        f'<p class="lede">{lede}</p></header>'
        f'<div class="article-body">{paragraphs}'
        f'<div class="pullquote"><p>«{quote}»</p>'
        f"<p>— {expert}, {quote_role}</p></div></div>"
        f'<ul class="tags"><li class="tag-head">{tags_head}</li>{tags}</ul>'
        f'<div class="related"><h2>{related_head}</h2><ul>{related}</ul></div>'
        "</article>"
    )
    target: dict[str, Any] = {
        "title": title,
        "author": author,
        "published_at": published,
    }
    if complexity >= 3:
        target["modified_at"] = modified
    if complexity >= 2:
        target["section"] = section
        target["description"] = lede
    jsonld = {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": title,
        "author": {"@type": "Person", "name": author},
        "datePublished": published,
        "dateModified": modified,
        "articleSection": section,
        "description": lede,
    }
    return Candidate(
        category="article",
        lang=lang,
        idx=idx,
        title=title,
        target=target,
        core=core,
        extras=[],
        filler=list(ART_FILLER[lang]),
        complexity=complexity,
        site=SITES[("article", lang)],
        slug=f"article-{idx:04d}",
        jsonld=jsonld,
        instruction=INSTRUCTIONS["article"][lang][(idx - 1) % 5],
    )


def listing_candidate(lang: str, idx: int, rng: random.Random) -> Candidate:
    theme = LISTING_THEMES[(idx - 1) % len(LISTING_THEMES)]
    variant = (idx - 1) // len(LISTING_THEMES)
    complexity = _complexity(rng)
    count = complexity + 2
    city = CITIES[lang][variant % len(CITIES[lang])]
    currency = "RUB" if lang == "ru" else "GBP"
    chosen = [theme["items"][(variant + j) % len(theme["items"])] for j in range(count)]
    listing_rows: list[str] = []
    items: list[dict[str, Any]] = []
    for j, pair in enumerate(chosen):
        name = pair[0] if lang == "ru" else pair[1]
        price = int(theme["base_" + lang] + variant * (37 + j) + j * 113)
        price_text = format_price(price, lang, currency)
        url = f"https://market.example/{lang}/item/{idx}-{j + 1}"
        condition = CONDITIONS[lang][(idx + j) % len(CONDITIONS[lang])]
        if lang == "ru":
            snippet = f"{condition.capitalize()}, самовывоз или доставка по городу."
        else:
            snippet = f"{condition.capitalize()}, pickup or local delivery available."
        listing_rows.append(
            '<li class="listing-item">'
            f'<a class="listing-title" href="{url}">{name}</a>'
            f'<span class="price">{price_text}</span>'
            f'<p class="snippet">{snippet}</p>'
            f'<p class="meta">{city} · {human_date(DATES[(idx + j) % len(DATES)], lang)}</p>'
            "</li>"
        )
        items.append({"title": name, "url": url, "price": price, "currency": currency})
    theme_name = theme[lang]
    if lang == "ru":
        title = f"Объявления: {theme_name} — {city}"
        summary = (
            f"Найдено {count} объявлений в городе {city}. Все цены указаны в {currency}."
        )
        sort = "Сортировать: по дате"
    else:
        title = f"Listings: {theme_name} — {city}"
        summary = f"Found {count} listings in {city}. All prices are in {currency}."
        sort = "Sort by: date"
    core = (
        f'<div class="listing-page"><h1>{title}</h1><p class="summary">{summary}</p>'
        f'<div class="toolbar"><span>{sort}</span><span>1 / 3</span></div>'
        f'<ul class="listing">{"".join(listing_rows)}</ul></div>'
    )
    if lang == "ru":
        extra1 = (
            '<div class="pagination"><a href="/page/1">1</a>'
            '<a href="/page/2">2</a><a href="/page/3">3</a></div>'
        )
        extra2 = (
            '<section class="queries"><h2>Популярные запросы</h2><ul>'
            f"<li>{theme_name} {city}</li><li>{theme_name} недорого</li>"
            "<li>доставка сегодня</li></ul></section>"
        )
        extra3 = (
            '<section class="safety"><h2>Безопасная сделка</h2>'
            "<p>Проверяйте товар при встрече и не переводите предоплату незнакомым "
            "продавцам.</p></section>"
        )
    else:
        extra1 = (
            '<div class="pagination"><a href="/page/1">1</a>'
            '<a href="/page/2">2</a><a href="/page/3">3</a></div>'
        )
        extra2 = (
            '<section class="queries"><h2>Popular searches</h2><ul>'
            f"<li>{theme_name} {city}</li><li>cheap {theme_name.lower()}</li>"
            "<li>same-day delivery</li></ul></section>"
        )
        extra3 = (
            '<section class="safety"><h2>Safe trading</h2>'
            "<p>Inspect the item in person and never send a deposit to an unknown "
            "seller.</p></section>"
        )
    extras = [extra1]
    if complexity >= 2:
        extras.append(extra2)
    if complexity >= 3:
        extras.append(extra3)
    jsonld = {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "numberOfItems": count,
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": n,
                "item": {
                    "@type": "Product",
                    "name": item["title"],
                    "url": item["url"],
                    "offers": {
                        "price": item["price"],
                        "priceCurrency": item["currency"],
                    },
                },
            }
            for n, item in enumerate(items, start=1)
        ],
    }
    return Candidate(
        category="listing",
        lang=lang,
        idx=idx,
        title=title,
        target={"count": count, "items": items},
        core=core,
        extras=extras,
        filler=list(LISTING_FILLER[lang]),
        complexity=complexity,
        site=SITES[("listing", lang)],
        slug=f"listing-{idx:04d}",
        jsonld=jsonld,
        instruction=INSTRUCTIONS["listing"][lang][(idx - 1) % 5],
    )


def job_candidate(lang: str, idx: int, rng: random.Random) -> Candidate:
    role, base_salary = JOB_ROLES[lang][(idx - 1) % len(JOB_ROLES[lang])]
    company = COMPANIES[lang][(idx * 3) % len(COMPANIES[lang])]
    city = JOB_CITIES[lang][(idx * 7) % len(JOB_CITIES[lang])]
    code, ru_employment, en_employment = EMPLOYMENT[idx % len(EMPLOYMENT)]
    employment_label = ru_employment if lang == "ru" else en_employment
    complexity = _complexity(rng)
    salary = int(base_salary + (idx % 5) * (5000 if lang == "ru" else 150))
    currency = "RUB" if lang == "ru" else "USD"
    posted = DATES[(idx * 5) % len(DATES)]
    salary_text = format_price(salary, lang, currency)
    if lang == "ru":
        title = role
        salary_line = f'<p class="salary">{salary_text}</p>'
        meta = f'<p class="meta">{city} · {employment_label} ({code})</p>'
        posted_line = f'<p class="posted">Опубликовано: {posted}</p>'
        dl = (
            f"<dt>Компания</dt><dd>{company}</dd>"
            f"<dt>Город</dt><dd>{city}</dd>"
            f"<dt>Зарплата</dt><dd>{salary_text}</dd>"
            f"<dt>Валюта</dt><dd>{currency}</dd>"
            f"<dt>Тип занятости</dt><dd>{code}</dd>"
            f"<dt>Номер вакансии</dt><dd>JOB-{idx:05d}</dd>"
        )
        desc_head = "О вакансии"
        resp_head = "Чем предстоит заниматься"
        req_head = "Требования"
        benefits_head = "Условия"
        description = (
            f"{company} ищет специалиста на позицию «{role}». Мы предлагаем стабильную "
            "загрузку, понятные задачи и поддержку на старте."
        )
        responsibilities = (
            "<li>Планировать задачи и согласовывать сроки с командой</li>"
            "<li>Следить за качеством результата и документацией</li>"
            "<li>Участвовать в еженедельных встречах и обсуждениях</li>"
        )
        requirements = (
            "<li>Опыт работы по направлению от одного года</li>"
            "<li>Умение работать в команде и объяснять решения</li>"
            "<li>Внимательность к деталям и ответственность</li>"
        )
        benefits = (
            "<li>Официальное трудоустройство и оплачиваемый отпуск</li>"
            "<li>Обучение за счёт работодателя</li>"
            "<li>Гибридный формат по согласованию</li>"
        )
    else:
        title = role
        salary_line = f'<p class="salary">{salary_text}</p>'
        meta = f'<p class="meta">{city} · {employment_label} ({code})</p>'
        posted_line = f'<p class="posted">Posted: {posted}</p>'
        dl = (
            f"<dt>Company</dt><dd>{company}</dd>"
            f"<dt>City</dt><dd>{city}</dd>"
            f"<dt>Salary</dt><dd>{salary_text}</dd>"
            f"<dt>Currency</dt><dd>{currency}</dd>"
            f"<dt>Employment type</dt><dd>{code}</dd>"
            f"<dt>Vacancy no.</dt><dd>JOB-{idx:05d}</dd>"
        )
        desc_head = "About the role"
        resp_head = "What you will do"
        req_head = "Requirements"
        benefits_head = "Benefits"
        description = (
            f"{company} is hiring a {role}. We offer a steady workload, clear tasks and "
            "support during onboarding."
        )
        responsibilities = (
            "<li>Plan tasks and agree timelines with the team</li>"
            "<li>Own quality and keep documentation up to date</li>"
            "<li>Join weekly meetings and planning sessions</li>"
        )
        requirements = (
            "<li>At least one year of relevant experience</li>"
            "<li>Strong communication and teamwork skills</li>"
            "<li>Attention to detail and a responsible approach</li>"
        )
        benefits = (
            "<li>Permanent contract and paid holiday</li>"
            "<li>Employer-funded training</li>"
            "<li>Hybrid working by agreement</li>"
        )
    core = (
        '<div class="job-page"><div class="job-header">'
        f'<p class="company">{company}</p><h1>{title}</h1>{salary_line}{meta}{posted_line}'
        f'<dl class="job-facts">{dl}</dl></div>'
        f'<section class="job-description"><h2>{desc_head}</h2><p>{description}</p></section>'
        "</div>"
    )
    extras = [
        f'<section class="duties"><h2>{resp_head}</h2><ul>{responsibilities}</ul></section>',
    ]
    if complexity >= 2:
        extras.append(
            f'<section class="requirements"><h2>{req_head}</h2><ul>{requirements}</ul></section>'
        )
    if complexity >= 3:
        extras.append(
            f'<section class="benefits"><h2>{benefits_head}</h2><ul>{benefits}</ul></section>'
        )
    target: dict[str, Any] = {
        "title": title,
        "company": company,
        "salary": salary,
        "currency": currency,
    }
    if complexity >= 2:
        target["location"] = city
        target["posted_at"] = posted
    if complexity >= 3:
        target["employment_type"] = code
    jsonld = {
        "@context": "https://schema.org",
        "@type": "JobPosting",
        "title": title,
        "hiringOrganization": {"@type": "Organization", "name": company},
        "jobLocation": {
            "@type": "Place",
            "address": {"@type": "PostalAddress", "addressLocality": city},
        },
        "baseSalary": {
            "@type": "MonetaryAmount",
            "currency": currency,
            "value": {"@type": "QuantitativeValue", "value": salary},
        },
        "datePosted": posted,
        "employmentType": code,
    }
    return Candidate(
        category="job",
        lang=lang,
        idx=idx,
        title=title,
        target=target,
        core=core,
        extras=extras,
        filler=list(JOB_FILLER[lang]),
        complexity=complexity,
        site=SITES[("job", lang)],
        slug=f"job-{idx:04d}",
        jsonld=jsonld,
        instruction=INSTRUCTIONS["job"][lang][(idx - 1) % 5],
    )


ART_FILLER = {
    "ru": (
        "Материал подготовлен редакцией; при цитировании ссылка на источник "
        "обязательна. Ошибки и уточнения присылайте на адрес редакции.",
        "Корреспонденты собирают комментарии жителей в течение дня, поэтому "
        "материал может обновляться. Следите за версией новости на этой странице.",
        "Ранее мы писали о других изменениях в городе: подписывайтесь на наш "
        "telegram-канал, чтобы не пропустить важные новости района.",
    ),
    "en": (
        "This material was prepared by the editorial team; a link back is "
        "required when quoting. Send corrections and questions to the newsroom.",
        "Reporters are collecting resident comments throughout the day, so the "
        "story may be updated. Check the version stamp on this page.",
        "We previously covered other changes in the city: subscribe to our "
        "channel to keep up with district news.",
    ),
}

def realestate_candidate(lang: str, idx: int, rng: random.Random) -> Candidate:
    street = REALESTATE_STREETS[lang][(idx - 1) % len(REALESTATE_STREETS[lang])]
    district = REALESTATE_DISTRICTS[lang][(idx - 1) % len(REALESTATE_DISTRICTS[lang])]
    complexity = _complexity(rng)
    rooms = 1 + idx % 4
    area = round(28.0 + (idx * 3.7) % 70 + (idx % 10) / 10.0, 1)
    floor = 1 + idx % 9
    floors = max(5, floor + idx % 6)
    per_sqm = PRICE_PER_SQM[lang][idx % len(PRICE_PER_SQM[lang])]
    price = int(round(area * per_sqm / 1000.0)) * 1000
    currency = "RUB" if lang == "ru" else "GBP"
    house = 2 + (idx * 5) % 60
    if lang == "ru":
        area_text = f"{area}".replace(".", ",")
        title = f"Квартира {rooms} комн. {area_text} м², ул. {street}, {house}"
        address = f"ул. {street}, д. {house}, р-н {district}"
        price_text = format_price(price, lang, currency)
        sqm_text = format_price(per_sqm, lang, currency)
        desc_head = "Описание"
        desc = (
            f"Светлая квартира в районе {district}: {rooms} комнаты, площадь "
            f"{area} м², {floor}-й этаж из {floors}. Окна выходят во двор, "
            "ремонт сделан три года назад, мебель остаётся по договорённости."
        )
        params_head = "Параметры"
        table_rows = (
            f"<tr><th>Комнаты</th><td>{rooms}</td></tr>"
            f"<tr><th>Площадь</th><td>{area_text} м²</td></tr>"
            f"<tr><th>Этаж</th><td>{floor} из {floors}</td></tr>"
            f"<tr><th>Цена за м²</th><td>{sqm_text}</td></tr>"
        )
        nearby_head = "Рядом"
        nearby = (
            "<li>Метро — 8 минут пешком</li><li>Школа и детский сад во дворе</li>"
            "<li>Продуктовый магазин на первом этаже</li>"
        )
        addr_label = "Адрес"
        area_label = "Площадь"
        rooms_label = "Комнаты"
        floor_label = "Этаж"
        cur_label = "Валюта"
    else:
        area_text = f"{area}"
        title = f"{rooms}-bed flat {area_text} m², {house} {street}"
        address = f"{house} {street}, {district}"
        price_text = format_price(price, lang, currency)
        sqm_text = format_price(per_sqm, lang, currency)
        desc_head = "Description"
        desc = (
            f"A bright flat in {district}: {rooms} bedrooms, {area} m² of floor "
            f"space, floor {floor} of {floors}. Windows face the courtyard, the "
            "renovation is three years old and furniture is included by agreement."
        )
        params_head = "Details"
        table_rows = (
            f"<tr><th>Bedrooms</th><td>{rooms}</td></tr>"
            f"<tr><th>Floor area</th><td>{area} m²</td></tr>"
            f"<tr><th>Floor</th><td>{floor} of {floors}</td></tr>"
            f"<tr><th>Price per m²</th><td>{sqm_text}</td></tr>"
        )
        nearby_head = "Nearby"
        nearby = (
            "<li>Tube station — 8 minutes on foot</li>"
            "<li>School and nursery in the courtyard</li>"
            "<li>Convenience store on the ground floor</li>"
        )
        addr_label = "Address"
        area_label = "Floor area"
        rooms_label = "Bedrooms"
        floor_label = "Floor"
        cur_label = "Currency"
    m2 = "м²" if lang == "ru" else "m²"
    floor_text = (
        f"{floor} из {floors}" if lang == "ru" else f"{floor} of {floors}"
    )
    crumbs_home = "Главная" if lang == "ru" else "Home"
    crumbs_cat = "Недвижимость" if lang == "ru" else "Property"
    badge = "Топ" if lang == "ru" else "Top"
    core = (
        f'<div class="breadcrumbs"><a href="/">{crumbs_home}</a>'
        f' / <a href="/realty">{crumbs_cat}</a> / <span>{district}</span></div>'
        '<div class="estate-layout"><div class="estate-gallery">'
        f'<img src="/img/estate-{idx:04d}.jpg" alt="{title}">'
        f'<span class="badge">{badge}</span></div>'
        '<div class="estate-summary">'
        f"<h1>{title}</h1>"
        f'<div class="price-row"><span class="price">{price_text}</span>'
        f'<span class="price-per">{sqm_text} / {m2}</span></div>'
        f'<dl class="key-facts"><dt>{addr_label}</dt><dd>{address}</dd>'
        f"<dt>{area_label}</dt><dd>{area} {m2}</dd>"
        f"<dt>{rooms_label}</dt><dd>{rooms}</dd>"
        f"<dt>{floor_label}</dt><dd>{floor_text}</dd>"
        f"<dt>{cur_label}</dt><dd>{currency}</dd></dl>"
        "</div></div>"
    )
    extras = [
        f'<section class="description"><h2>{desc_head}</h2><p>{desc}</p></section>',
    ]
    if complexity >= 2:
        extras.append(
            f'<section class="params"><h2>{params_head}</h2>'
            f"<table><tbody>{table_rows}</tbody></table></section>"
        )
    if complexity >= 3:
        extras.append(
            f'<section class="nearby"><h2>{nearby_head}</h2><ul>{nearby}</ul></section>'
        )
    target: dict[str, Any] = {
        "title": title,
        "price": price,
        "currency": currency,
        "address": address,
        "area": area,
        "rooms": rooms,
    }
    jsonld = {
        "@context": "https://schema.org",
        "@type": "RealEstateListing",
        "name": title,
        "offers": {"@type": "Offer", "price": price, "priceCurrency": currency},
        "address": {"@type": "PostalAddress", "streetAddress": address},
        "floorSize": {"@type": "QuantitativeValue", "value": area, "unitCode": "MTK"},
        "numberOfRooms": rooms,
    }
    return Candidate(
        category="realestate",
        lang=lang,
        idx=idx,
        title=title,
        target=target,
        core=core,
        extras=extras,
        filler=list(REALESTATE_FILLER[lang]),
        complexity=complexity,
        site=SITES[("realestate", lang)],
        slug=f"estate-{idx:04d}",
        jsonld=jsonld,
        instruction=INSTRUCTIONS["realestate"][lang][(idx - 1) % 5],
    )


BUILDERS = {
    "product": product_candidate,
    "article": article_candidate,
    "listing": listing_candidate,
    "job": job_candidate,
    "realestate": realestate_candidate,
}


def visible_text(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = html_lib.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def absent_values(target: Any, html: str) -> list[str]:
    """Return dotted paths of target leaves that are absent from the page."""

    digits = re.sub(r"\D", "", html)
    lowered = html.lower()
    missing: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, int):
            if str(node) not in digits:
                missing.append(path)
        elif isinstance(node, float):
            variants = (str(node), str(node).replace(".", ","))
            if not any(variant in html for variant in variants):
                missing.append(path)
        elif isinstance(node, str) and node.lower() not in lowered:
            missing.append(path)

    walk(target, "")
    return missing


def anchor_missing(target: Any, html: str) -> bool:
    text = visible_text(html).lower()
    anchor = target.get("title") if isinstance(target, dict) else None
    if not anchor and isinstance(target, dict) and target.get("items"):
        anchor = target["items"][0].get("title")
    return bool(anchor) and str(anchor)[:80].lower() not in text


def build_record(
    cand: Candidate, seen: set[str]
) -> tuple[dict[str, Any] | None, str | None, int]:
    """Render one candidate; return (record, reject_code, attempts_used)."""

    attempts = 0
    for round_ in range(1, 6):
        attempts = round_
        html, compressed = build_page(cand)
        if len(compressed) >= MIN_CHARS or not cand.filler:
            break
        cand.filler = cand.filler + list(cand.filler)
    if not (MIN_CHARS <= len(compressed) <= MAX_CHARS):
        return None, "E_HTML_LENGTH", attempts
    if anchor_missing(cand.target, compressed):
        return None, "E_ANCHOR_NOT_VISIBLE", attempts
    absent = absent_values(cand.target, compressed)
    if absent:
        return None, f"E_VALUE_ABSENT:{absent[0]}", attempts
    digest = hashlib.blake2b(compressed.encode("utf-8"), digest_size=16).hexdigest()
    if digest in seen:
        return None, "E_DUPLICATE", attempts
    record = {
        "id": make_id("synth", cand.category, cand.lang, cand.idx),
        "group_id": f"synthetic.{cand.category}.{cand.lang}",
        "task": "extract",
        "instruction": cand.instruction,
        "input_html": compressed,
        "target": cand.target,
        "meta": {
            "lang": cand.lang,
            "complexity": cand.complexity,
            "source_url": f"https://synthetic.local/{cand.category}/{cand.lang}/{cand.idx}",
            "n_tokens": None,
            "pipeline_version": PIPELINE_VERSION,
        },
    }
    seen.add(digest)
    return record, None, attempts


def quotas(count: int) -> dict[str, int]:
    base, rem = divmod(count, len(CATEGORIES))
    return {
        cat: base + (1 if i < rem else 0) for i, cat in enumerate(CATEGORIES)
    }


def plan_buckets(count: int, seed: int) -> list[tuple[str, str, int]]:
    """Return (category, lang, quota) buckets with an exact 60/40 ru/en split."""

    sched = random.Random(seed ^ 0x5EED)
    plan: list[tuple[str, str, int]] = []
    for cat, quota in quotas(count).items():
        ru_quota = (quota * 3 + 2) // 5
        langs = ["ru"] * ru_quota + ["en"] * (quota - ru_quota)
        sched.shuffle(langs)
        for lang in langs:
            plan.append((cat, lang, 1))
    return plan


def generate(out_dir: Path, count: int, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    buckets = plan_buckets(count, seed)
    quota: Counter[tuple[str, str]] = Counter()
    filled: Counter[tuple[str, str]] = Counter()
    for cat, lang, _ in buckets:
        quota[(cat, lang)] += 1

    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    records: list[dict] = []
    per_bucket = quota.most_common(1)[0][1] if quota else 0

    for cat, lang, _ in buckets:
        key = (cat, lang)
        idx = 1
        while filled[key] < quota[key]:
            if idx > quota[key] * 60 + 100:
                raise RuntimeError(
                    f"cannot fill bucket {cat}/{lang}: {filled[key]}/{quota[key]}"
                )
            cand = BUILDERS[cat](lang, idx, rng)
            idx += 1
            record, code, _attempts = build_record(cand, seen)
            if record is None:
                rejected[code or "E_UNKNOWN"] += 1
                continue
            try:
                Sample.model_validate(record)
            except Exception as exc:  # noqa: BLE001 - report and retry with next idx
                seen.discard(
                    hashlib.blake2b(
                        record["input_html"].encode("utf-8"), digest_size=16
                    ).hexdigest()
                )
                rejected[f"E_SCHEMA:{type(exc).__name__}"] += 1
                continue
            records.append(record)
            filled[key] += 1

    assert sum(filled.values()) == len(records), "bucket accounting mismatch"
    stats = {
        "seed": seed,
        "requested": count,
        "written": len(records),
        "per_bucket_max": per_bucket,
        "rejected": dict(rejected),
    }
    return records, stats


def write_shards(records: list[dict], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cat: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
    for record in records:
        cat = str(record["group_id"]).split(".")[1]
        by_cat[cat].append(record)
    counts: dict[str, int] = {}
    for cat, rows in by_cat.items():
        path = out_dir / f"{cat}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[cat] = len(rows)
    return counts


def token_report(records: list[dict]) -> tuple[dict, str | None]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        tokenizer = load_tokenizer(TOKENIZER_NAME)
    except Exception as exc:  # noqa: BLE001 - offline runs may lack the cache
        return (
            {"p50": None, "p95": None, "max": None,
             "note": f"tokenizer unavailable: {type(exc).__name__}: {exc}"},
            None,
        )
    samples = [Sample.model_validate(record) for record in records]
    lengths = token_lengths(samples, tokenizer)
    for record, length in zip(records, lengths, strict=True):
        record["meta"]["n_tokens"] = length
    stats = length_stats(lengths)
    return (
        {"p50": stats["p50"], "p95": stats["p95"], "max": stats["max"], "note": None},
        TOKENIZER_NAME,
    )


def build_report(
    records: list[dict],
    gen_stats: dict,
    counts: dict[str, int],
    tokens: dict,
    tokenizer_name: str | None,
    duration: float,
    files: list[str],
) -> dict:
    langs = Counter(record["meta"]["lang"] for record in records)
    complexity = Counter(record["meta"]["complexity"] for record in records)
    lengths = sorted(len(record["input_html"]) for record in records)
    total = len(records)
    html_stats = length_stats(lengths)
    return {
        "seed": gen_stats["seed"],
        "requested": gen_stats["requested"],
        "written": gen_stats["written"],
        "by_category": counts,
        "by_lang": dict(langs),
        "lang_share": {
            lang: round(langs.get(lang, 0) / total, 4) for lang in LANGS
        },
        "by_complexity": {str(k): complexity[k] for k in sorted(complexity)},
        "html_chars": {
            "min": lengths[0],
            "p50": html_stats["p50"],
            "p95": html_stats["p95"],
            "max": html_stats["max"],
        },
        "token_stats": tokens,
        "tokenizer": tokenizer_name,
        "rejected": gen_stats["rejected"],
        "files": files,
        "duration_s": round(duration, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Детерминированный офлайн-синтез HTML→JSON датасета web2spec"
    )
    parser.add_argument("--out", default="data/raw/synthetic")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stats", default="data/reports/synthetic_stats.json")
    args = parser.parse_args(argv)
    if args.count < len(CATEGORIES):
        parser.error(f"--count must be at least {len(CATEGORIES)}")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    stats_path = Path(args.stats)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path

    started = time.perf_counter()
    records, gen_stats = generate(out_dir, args.count, args.seed)
    tokens, tokenizer_name = token_report(records)
    if tokenizer_name:
        # n_tokens changed after the first validation pass: re-validate.
        for record in records:
            Sample.model_validate(record)
    counts = write_shards(records, out_dir)
    duration = time.perf_counter() - started

    try:
        base = out_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        base = out_dir.as_posix()
    files = [f"{base}/{cat}.jsonl" for cat in CATEGORIES]
    report = build_report(
        records, gen_stats, counts, tokens, tokenizer_name, duration, files
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if sum(counts.values()) == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
