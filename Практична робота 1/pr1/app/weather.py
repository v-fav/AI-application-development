"""Модуль інтеграції із зовнішнім API погоди.

Це єдине місце застосунку, яке знає про HTTP: адреси сервісів, параметри
запиту, коди відповіді й формат JSON. Веб-рівень (`app/main.py`) отримує
звідси готовий результат або зрозумілу помилку і нічого не знає про
`requests`.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі рішення
з розділу 4 практичної роботи:

* як передати параметри запиту, не склеюючи URL вручну;
* яке обмеження часу (timeout) поставити й що робити, коли воно спрацювало;
* чи однаково реагувати на помилку клієнта (4xx) і сервера (5xx);
* як повестися, коли міста не знайдено або у відповіді немає потрібних полів;
* що саме віддавати назовні при успіху і як позначати помилку.

Реальні відповіді обох сервісів збережено в папці `samples/` — подивіться їх
перед тим, як писати розбір відповіді.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

TIMEOUT_SECONDS = 10

class WeatherError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    """Помилка отримання погоди, зрозуміла веб-рівню.

    Заготовка. Вирішіть, чи достатньо одного типу помилки, чи їх варто
    розрізняти — місто не знайдено, сервіс недоступний, відповідь не та,
    якої очікували. Від цього залежить, який HTTP-статус поверне застосунок
    і що побачить користувач.
    """


def find_city(name: str):
    if not name.strip():
        raise WeatherError(
            "Назва міста не може бути порожньою.",
            status_code=400
        )

    params = {
        "name": name.strip(),
        "count": 1,
        "language": "uk",
        "format": "json",
    }

    try:
        response = requests.get(
            GEOCODING_URL,
            params=params,
            timeout=TIMEOUT_SECONDS
        )
    except requests.Timeout:
        raise WeatherError(
            "Сервіс геокодування не відповів вчасно.",
            status_code=504
        )
    except requests.RequestException:
        raise WeatherError(
            "Не вдалося підключитися до сервісу геокодування.",
            status_code=503
        )

    if 400 <= response.status_code < 500:
        raise WeatherError(
            "Сервіс геокодування відхилив запит.",
            status_code=400
        )

    if 500 <= response.status_code:
        raise WeatherError(
            "Сервіс геокодування тимчасово недоступний.",
            status_code=503
        )

    try:
        data = response.json()
    except ValueError:
        raise WeatherError(
            "Сервіс геокодування повернув некоректний JSON.",
            status_code=502
        )

    if not isinstance(data, dict):
        raise WeatherError(
            "Сервіс геокодування повернув неочікуваний формат даних.",
            status_code=502
        )

    results = data.get("results")

    if not isinstance(results, list) or not results:
        raise WeatherError(
            "Місто не знайдено.",
            status_code=404
        )

    city = results[0]

    if not isinstance(city, dict):
        raise WeatherError(
            "Сервіс геокодування повернув неочікувані дані.",
            status_code=502
        )

    latitude = city.get("latitude")
    longitude = city.get("longitude")

    if not isinstance(latitude, (int, float)) or not isinstance(
        longitude, (int, float)
    ):
        raise WeatherError(
            "У відповіді сервісу немає координат міста.",
            status_code=502
        )

    return {
        "name": city.get("name", name.strip()),
        "latitude": latitude,
        "longitude": longitude,
    }

    """Знайти координати міста за його назвою.

    Що саме повертати — вирішіть самі: пару чисел, словник, окремий тип.
    Врахуйте випадок, коли міста з такою назвою немає.
    """
    raise NotImplementedError("find_city ще не реалізовано")


def get_current_weather(city: str):
    location = find_city(city)

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": "temperature_2m,wind_speed_10m",
        "timezone": "auto",
    }

    try:
        response = requests.get(
            FORECAST_URL,
            params=params,
            timeout=TIMEOUT_SECONDS
        )
    except requests.Timeout:
        raise WeatherError(
            "Сервіс погоди не відповів вчасно.",
            status_code=504
        )
    except requests.RequestException:
        raise WeatherError(
            "Не вдалося підключитися до сервісу погоди.",
            status_code=503
        )

    if 400 <= response.status_code < 500:
        raise WeatherError(
            "Сервіс погоди відхилив запит.",
            status_code=400
        )

    if 500 <= response.status_code:
        raise WeatherError(
            "Сервіс погоди тимчасово недоступний.",
            status_code=503
        )

    try:
        data = response.json()
    except ValueError:
        raise WeatherError(
            "Сервіс погоди повернув некоректний JSON.",
            status_code=502
        )

    if not isinstance(data, dict):
        raise WeatherError(
            "Сервіс погоди повернув неочікуваний формат даних.",
            status_code=502
        )

    current = data.get("current")

    if not isinstance(current, dict):
        raise WeatherError(
            "У відповіді сервісу немає даних про поточну погоду.",
            status_code=502
        )

    temperature = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")

    if not isinstance(temperature, (int, float)) or not isinstance(
        wind_speed, (int, float)
    ):
        raise WeatherError(
            "У відповіді сервісу немає потрібних даних про погоду.",
            status_code=502
        )

    return {
        "city": location["name"],
        "temperature": temperature,
        "temperature_unit": data.get("current_units", {}).get(
            "temperature_2m", "°C"
        ),
        "wind_speed": wind_speed,
        "wind_speed_unit": data.get("current_units", {}).get(
            "wind_speed_10m", "km/h"
        ),
    }

    """Повернути поточну погоду в місті: температуру й швидкість вітру.

    Це функція, яку викликає веб-рівень. Вона поєднує геокодування і запит
    прогнозу та віддає результат у зручному для застосунку вигляді.
    """
    raise NotImplementedError("get_current_weather ще не реалізовано")
