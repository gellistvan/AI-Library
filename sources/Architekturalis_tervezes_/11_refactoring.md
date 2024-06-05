\newpage

## 11. Refaktorizálás

A refaktorizálás a szoftverfejlesztés azon fontos folyamata, amelynek célja a meglévő kód átszervezése és javítása anélkül, hogy a külső viselkedése megváltozna. Ez a fejezet a kód tisztításának és optimalizálásának módszereit vizsgálja, valamint bemutatja a legfontosabb refaktorizálási mintákat, amelyek segítenek a kód karbantarthatóságának, olvashatóságának és teljesítményének növelésében. A refaktorizálás révén a fejlesztők képesek eltávolítani a kódban felhalmozódott technikai adósságokat, minimalizálni a hibákat, és előkészíteni a rendszert a jövőbeli fejlesztésekre. Ebben a fejezetben megismerkedünk a refaktorizálás alapelveivel, technikáival és gyakorlati példákkal, amelyek segítségével hatékonyabb és megbízhatóbb szoftvereket hozhatunk létre.

### Kód tisztítása és optimalizálása

A kód tisztítása és optimalizálása alapvető lépés a szoftverfejlesztésben, amelynek célja, hogy a kód olvashatóbbá, karbantarthatóbbá és hatékonyabbá váljon. A tiszta kód könnyebben érthető és módosítható, ami csökkenti a hibák lehetőségét és megkönnyíti a jövőbeni fejlesztéseket. Az optimalizálás pedig biztosítja, hogy a kód a lehető legjobb teljesítményt nyújtsa. Ebben az alfejezetben részletesen megvizsgáljuk a kód tisztításának és optimalizálásának legfontosabb technikáit és gyakorlatát, valamint gyakorlati példákat mutatunk be ezek alkalmazására.

#### Kód tisztítása

1. **Nevek jelentése és konvenciók követése**: A változók, függvények és osztályok elnevezése legyen beszédes és kövesse az elfogadott konvenciókat. A jó elnevezések segítenek megérteni a kód célját és működését.

   **Példa**:
   ```python
   # Rossz példa
   def calc(x, y):
       return x * y

   # Jó példa
   def calculate_area(width, height):
       return width * height
   ```

2. **Kommentárok és dokumentáció**: A kódhoz írt kommentárok és dokumentáció segítenek megérteni a bonyolult logikákat és algoritmusokat. A dokumentáció különösen fontos a nyilvános API-k és könyvtárak esetében.

   **Példa**:
   ```python
   def calculate_area(width, height):
       """
       Calculate the area of a rectangle.
       :param width: The width of the rectangle
       :param height: The height of the rectangle
       :return: The area of the rectangle
       """
       return width * height
   ```

3. **Kód duplikáció megszüntetése**: A kódismétlés (DRY - Don't Repeat Yourself) elkerülése érdekében a közös logikát helyezzük át külön függvényekbe vagy modulokba. Ez megkönnyíti a kód karbantartását és csökkenti a hibalehetőségeket.

   **Példa**:
   ```python
   # Rossz példa
   def print_user_info(name, age):
       print(f"Name: {name}")
       print(f"Age: {age}")

   def print_product_info(product_name, price):
       print(f"Product: {product_name}")
       print(f"Price: {price}")

   # Jó példa
   def print_info(label, value):
       print(f"{label}: {value}")

   def print_user_info(name, age):
       print_info("Name", name)
       print_info("Age", age)

   def print_product_info(product_name, price):
       print_info("Product", product_name)
       print_info("Price", price)
   ```

4. **Függvények és osztályok egyszerűsítése**: A függvények és osztályok legyenek kicsik és egyszerűek, mindegyik csak egy feladatot lásson el. Az egyszerűsítés érdekében a nagyobb függvényeket bontsuk kisebb, önálló egységekre.

   **Példa**:
   ```python
   # Rossz példa
   def process_order(order):
       # Validate order
       if not validate_order(order):
           return "Invalid order"
       # Calculate total
       total = calculate_total(order)
       # Apply discount
       total = apply_discount(order, total)
       # Process payment
       if not process_payment(order, total):
           return "Payment failed"
       return "Order processed"

   # Jó példa
   def process_order(order):
       if not validate_order(order):
           return "Invalid order"
       total = calculate_total(order)
       total = apply_discount(order, total)
       if not process_payment(order, total):
           return "Payment failed"
       return "Order processed"

   def validate_order(order):
       # Order validation logic
       pass

   def calculate_total(order):
       # Total calculation logic
       pass

   def apply_discount(order, total):
       # Discount application logic
       pass

   def process_payment(order, total):
       # Payment processing logic
       pass
   ```

5. **Kód formázása és stílus**: A kód formázása legyen következetes, használjuk a megfelelő behúzásokat, sorvégeket és üres sorokat. Az egységes stílus megkönnyíti a kód olvasását és karbantartását.

   **Példa**:
   ```python
   # Rossz példa
   def foo():print("Hello")    print("World")

   # Jó példa
   def foo():
       print("Hello")
       print("World")
   ```

#### Optimalizálás

1. **Algoritmusok és adatszerkezetek optimalizálása**: A megfelelő algoritmusok és adatszerkezetek kiválasztása alapvetően befolyásolja a kód teljesítményét. Használjuk a leghatékonyabb algoritmusokat és adatszerkezeteket a feladat megoldására.

   **Példa**:
   ```python
   # Rossz példa: Lineáris keresés
   def linear_search(arr, x):
       for i in range(len(arr)):
           if arr[i] == x:
               return i
       return -1

   # Jó példa: Bináris keresés (feltételezi, hogy az arr rendezett)
   def binary_search(arr, x):
       left, right = 0, len(arr) - 1
       while left <= right:
           mid = (left + right) // 2
           if arr[mid] == x:
               return mid
           elif arr[mid] < x:
               left = mid + 1
           else:
               right = mid - 1
       return -1
   ```

2. **Hurokoptimalizálás**: Az ismétlődő műveletek és ciklusok optimalizálása jelentős teljesítménynövekedést eredményezhet. Kerüljük a felesleges számításokat és minimalizáljuk a ciklusok belsejében végzett műveletek számát.

   **Példa**:
   ```python
   # Rossz példa
   def compute_squares(nums):
       result = []
       for num in nums:
           result.append(num * num)
       return result

   # Jó példa: List comprehension használata
   def compute_squares(nums):
       return [num * num for num in nums]
   ```

3. **Memóriahasználat optimalizálása**: A memóriahasználat optimalizálása csökkenti a memóriafoglalásokat és javítja a teljesítményt. Használjunk hatékony adatszerkezeteket és gondoskodjunk a memória megfelelő felszabadításáról.

   **Példa**:
   ```python
   # Rossz példa
   def create_large_list(n):
       result = []
       for i in range(n):
           result.append(i)
       return result

   # Jó példa: Generátor használata
   def create_large_list(n):
       return (i for i in range(n))
   ```

4. **I/O műveletek optimalizálása**: Az I/O műveletek gyakran a rendszer szűk keresztmetszetét jelentik. Optimalizáljuk az I/O műveleteket a párhuzamos feldolgozás, gyorsítótárazás és aszinkron műveletek használatával.

   **Példa**:
   ```python
   import aiohttp
   import asyncio

   # Rossz példa: Szinkron I/O műveletek
   def fetch_data(url):
       response = requests.get(url)
       return response.text

   # Jó példa: Aszinkron I/O műveletek
   async def fetch_data(session, url):
       async with session.get(url) as response:
           return await response.text()

   async def main():
       async with aiohttp.ClientSession() as session:
           html = await fetch_data(session, 'http://example.com')
           print(html)

   asyncio.run(main())
   ```

5. **Profilozás és teljesítménymérés**: A kód optimalizálása előtt és után is végezzünk profilozást és teljesítménymérést, hogy azonosítsuk a szűk keresztmetszeteket és mérjük az optimalizálás hatékonyságát. Használjunk profilozó eszközöket, mint például a cProfile vagy a Py-Spy.

   **Példa**:
   ```python
   import cProfile

   def compute_squares(nums):
       return [num * num for num in nums]

   if __name__ == "__main__":
    cProfile.run('compute_squares(range(100000))')
   ```

#### Következtetés

A kód tisztítása és optimalizálása alapvető fontosságú a szoftverfejlesztés során. A jó elnevezések, kommentárok, a kód duplikáció megszüntetése, a függvények és osztályok egyszerűsítése, valamint a következetes kódformázás mind hozzájárulnak a tiszta és karbantartható kódhoz. Az algoritmusok és adatszerkezetek optimalizálása, a hurokoptimalizálás, a memóriahasználat optimalizálása, az I/O műveletek optimalizálása és a profilozás segítenek a kód teljesítményének javításában. Ezek az elvek és technikák biztosítják, hogy a szoftver hatékony, gyors és könnyen karbantartható legyen, ami hosszú távon csökkenti a fejlesztési költségeket és növeli a felhasználói elégedettséget.

### Refaktorizálási minták

A refaktorizálás olyan módszerek és technikák összessége, amelyek segítségével a meglévő kódot újraszervezzük és javítjuk, anélkül, hogy a külső viselkedése megváltozna. A refaktorizálási minták olyan bevált megoldások, amelyek ismétlődő kódproblémákra kínálnak hatékony és megbízható megoldásokat. Ezek a minták segítenek a kód olvashatóságának, karbantarthatóságának és bővíthetőségének növelésében. Ebben az alfejezetben részletesen bemutatjuk a legfontosabb refaktorizálási mintákat és gyakorlati példákat adunk azok alkalmazására.

#### Extract Method (Metóduskivonás)

A metóduskivonás célja, hogy egy hosszú és bonyolult metódusból kisebb, önálló részeket különítsünk el, és külön metódusokba helyezzük őket. Ez növeli a kód olvashatóságát és újrahasználhatóságát.

**Példa**:
```python
# Eredeti kód

def print_owing():
    outstanding = 0.0

    # Print banner
    print("**************************")
    print("***** Customer Owes ******")
    print("**************************")

    # Calculate outstanding
    for order in orders:
        outstanding += order.amount

    # Print details
    print(f"name: {name}")
    print(f"amount: {outstanding}")

# Refaktorizált kód

def print_owing():
    print_banner()
    outstanding = calculate_outstanding()
    print_details(outstanding)

def print_banner():
    print("**************************")
    print("***** Customer Owes ******")
    print("**************************")

def calculate_outstanding():
    outstanding = 0.0
    for order in orders:
        outstanding += order.amount
    return outstanding

def print_details(outstanding):
    print(f"name: {name}")
    print(f"amount: {outstanding}")
```

#### Inline Method (Metódus inline-olása)

Az inline-olás az a folyamat, amikor egy metódus tartalmát visszahelyezzük a hívási helyére. Ez akkor hasznos, ha a metódus tartalma rövid és csak egyszer hívják meg, vagy ha a metódus feleslegessé vált.

**Példa**:
```python
# Eredeti kód

def get_rating():
    return more_than_five_late_deliveries()

def more_than_five_late_deliveries():
    return number_of_late_deliveries > 5

# Refaktorizált kód

def get_rating():
    return number_of_late_deliveries > 5
```

#### Replace Temp with Query (Ideiglenes változó cseréje lekérdezéssel)

Ez a minta az ideiglenes változók eltávolítását és helyettesítését javasolja olyan metódusokkal, amelyek közvetlenül visszaadják az értékeket. Ez növeli a kód olvashatóságát és egyszerűsíti a logikát.

**Példa**:
```python
# Eredeti kód

def calculate_total():
    base_price = quantity * item_price
    if base_price > 1000:
        return base_price * 0.95
    else:
        return base_price * 0.98

# Refaktorizált kód

def calculate_total():
    if base_price() > 1000:
        return base_price() * 0.95
    else:
        return base_price() * 0.98

def base_price():
    return quantity * item_price
```

#### Introduce Explaining Variable (Magyarázó változó bevezetése)

Ez a minta azt javasolja, hogy vezessünk be magyarázó változókat a komplex kifejezések egyszerűsítése és olvashatóságának javítása érdekében. Ez megkönnyíti a kód megértését és karbantartását.

**Példa**:
```python
# Eredeti kód

def price():
    return quantity * item_price - max(0, quantity - 500) * item_price * 0.05 + min(quantity * item_price * 0.1, 100)

# Refaktorizált kód

def price():
    base_price = quantity * item_price
    quantity_discount = max(0, quantity - 500) * item_price * 0.05
    shipping = min(base_price * 0.1, 100)
    return base_price - quantity_discount + shipping
```

#### Extract Class (Osztálykivonás)

Az osztálykivonás akkor hasznos, ha egy osztály túl sok felelősséget lát el. Az osztálykivonás során egy új osztályt hozunk létre, és az eredeti osztály felelősségeit és adattagjait áthelyezzük az új osztályba.

**Példa**:
```python
# Eredeti kód

class Person:
    def __init__(self, name, office_area_code, office_number):
        self.name = name
        self.office_area_code = office_area_code
        self.office_number = office_number

    def get_telephone_number(self):
        return f"({self.office_area_code}) {self.office_number}"

# Refaktorizált kód

class Person:
    def __init__(self, name, office_telephone):
        self.name = name
        self.office_telephone = office_telephone

    def get_telephone_number(self):
        return self.office_telephone.get_telephone_number()

class TelephoneNumber:
    def __init__(self, area_code, number):
        self.area_code = area_code
        self.number = number

    def get_telephone_number(self):
        return f"({self.area_code}) {self.number}"
```

#### Move Method (Metódus áthelyezése)

A metódus áthelyezése során egy metódust áthelyezünk egy másik osztályba, ahol jobban illeszkedik az osztály felelősségi körébe. Ez növeli a kód koherenciáját és csökkenti az osztályok közötti függőségeket.

**Példa**:
```python
# Eredeti kód

class Account:
    def __init__(self, account_type, days_overdrawn):
        self.account_type = account_type
        self.days_overdrawn = days_overdrawn

    def overdraft_charge(self):
        if self.account_type.is_premium:
            base_charge = 10
            if self.days_overdrawn <= 7:
                return base_charge
            else:
                return base_charge + (self.days_overdrawn - 7) * 0.85
        else:
            return self.days_overdrawn * 1.75

class AccountType:
    def __init__(self, is_premium):
        self.is_premium = is_premium

# Refaktorizált kód

class Account:
    def __init__(self, account_type, days_overdrawn):
        self.account_type = account_type
        self.days_overdrawn = days_overdrawn

    def overdraft_charge(self):
        return self.account_type.overdraft_charge(self.days_overdrawn)

class AccountType:
    def __init__(self, is_premium):
        self.is_premium = is_premium

    def overdraft_charge(self, days_overdrawn):
        if self.is_premium:
            base_charge = 10
            if days_overdrawn <= 7:
                return base_charge
            else:
                return base_charge + (days_overdrawn - 7) * 0.85
        else:
            return days_overdrawn * 1.75
```

#### Introduce Parameter Object (Paraméter objektum bevezetése)

Ez a minta azt javasolja, hogy ha egy metódus sok paramétert fogad, akkor ezeket a paramétereket egy objektumba csoportosítsuk. Ez csökkenti a metódus aláírásának hosszát és növeli a kód olvashatóságát.

**Példa**:
```python
# Eredeti kód

def book_flight(customer_name, customer_age, flight_number, departure, arrival):
    # Book flight logic
    pass

# Refaktorizált kód

class Customer:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Flight:
    def __init__(self, number, departure, arrival):
        self.number = number
        self.departure = departure
        self.arrival = arrival

def book_flight(customer, flight):
    # Book flight logic
    pass
```

#### Következtetés

A refaktorizálási minták alkalmazása elengedhetetlen a kód karbantarthatóságának és olvashatóságának növeléséhez. Az olyan minták, mint a metóduskivonás, metódus inline-olása, ideiglenes változó cseréje lekérdezéssel, magyarázó változó bevezetése, osztálykivonás, metódus áthelyezése és paraméter objektum bevezetése, mind hozzájárulnak a tisztább, érthetőbb és hatékonyabb kódhoz. Ezek a minták segítenek a kód komplexitásának csökkentésében, a kódbázis egységesítésében és a jövőbeli fejlesztések megkönnyítésében. Az ilyen refaktorizálási technikák rendszeres alkalmazása biztosítja, hogy a szoftver hosszú távon is megbízható és karbantartható maradjon.

