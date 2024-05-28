\newpage

## 6. Architektúra tervezés

Az architektúra tervezés a szoftverfejlesztés egyik legfontosabb lépése, amely meghatározza a rendszer alapvető szerkezetét és komponenseinek egymáshoz való viszonyát. E fejezet célja, hogy bemutassa a különböző architektúratípusokat, mint például a monolitikus, mikroszolgáltatások alapú és rétegezett architektúrákat, valamint részletesen tárgyalja az olyan architektúrális mintákat, mint az MVC, MVP és MVVM. Az architektúra tervezés alapelveinek és gyakorlatainak megértése elengedhetetlen a skálázható, karbantartható és rugalmas szoftverrendszerek létrehozásához, amelyek képesek megfelelni a változó üzleti igényeknek és technológiai kihívásoknak.

### Architektúratípusok

Az architektúratípusok meghatározása és kiválasztása kritikus lépés a szoftverfejlesztés folyamatában. A megfelelő architektúra kiválasztása alapvetően befolyásolja a rendszer teljesítményét, skálázhatóságát, karbantarthatóságát és rugalmasságát. Ebben az alfejezetben három fő architektúratípust tárgyalunk részletesen: a monolitikus architektúrát, a mikroszolgáltatások alapú architektúrát és a rétegezett architektúrát. Mindegyik architektúrának megvannak a maga előnyei és hátrányai, amelyek különböző alkalmazási területeken és kontextusokban mutatkoznak meg.

#### Monolitikus architektúra

A monolitikus architektúra egy hagyományos megközelítés, amelyben a teljes szoftverrendszer egyetlen, egységes egységként kerül megvalósításra. A monolitikus alkalmazásokban az összes komponens - beleértve az adatbáziskezelést, üzleti logikát és a felhasználói felületet - egyetlen kódbázison belül található. Ez az architektúra egyszerűsége miatt könnyen érthető és megvalósítható, különösen kisebb projektek esetében.

**Előnyök**:
- **Egyszerű fejlesztés és telepítés**: A monolitikus alkalmazások fejlesztése és telepítése egyszerűbb, mivel minden komponens egyetlen kódbázison belül van.
- **Egyszerűbb hibakeresés és tesztelés**: Mivel minden kód egy helyen található, a hibák nyomon követése és kijavítása, valamint a tesztelés is egyszerűbb lehet.
- **Egyszerűbb skálázás**: Az alkalmazás teljes egészében skálázható, ha szükséges, bár ez a megközelítés nem mindig a leghatékonyabb.

**Hátrányok**:
- **Korlátozott rugalmasság**: A monolitikus architektúrák nehezen alkalmazkodnak a változásokhoz, mivel egyetlen komponens módosítása az egész rendszer újratelepítését igényelheti.
- **Korlátozott skálázhatóság**: Bár a monolitikus alkalmazások skálázhatók, nem minden komponensnek van szüksége ugyanolyan skálázásra, ami erőforrás-pazarlást okozhat.
- **Nagyobb komplexitás a növekedéssel**: Ahogy az alkalmazás nő, a kód bonyolultsága is növekszik, ami nehezíti a karbantartást és a továbbfejlesztést.

**Példa**:
Egy egyszerű e-kereskedelmi alkalmazás, amely egyetlen kódbázisban tartalmazza a felhasználói regisztrációt, a termékkatalógust, a kosárkezelést és a rendelésfeldolgozást. Az alkalmazás könnyen fejleszthető és telepíthető, de nehezen skálázható és módosítható, ha az egyes komponensek különböző skálázási igényekkel rendelkeznek.

#### Mikroszolgáltatások alapú architektúra

A mikroszolgáltatások alapú architektúra egy modern megközelítés, amelyben az alkalmazás különálló, független szolgáltatások halmazából áll. Minden szolgáltatás saját adatbázissal és üzleti logikával rendelkezik, és gyakran különböző programozási nyelvekkel és technológiákkal valósítható meg. A szolgáltatások egymással jól definiált API-kon keresztül kommunikálnak, ami nagyobb rugalmasságot és skálázhatóságot biztosít.

**Előnyök**:
- **Rugalmasság és függetlenség**: A szolgáltatások függetlenül fejleszthetők, telepíthetők és skálázhatók, ami nagyobb rugalmasságot biztosít a fejlesztők számára.
- **Jobb hibakeresés és karbantartás**: A kisebb, jól definiált szolgáltatások könnyebben hibakereshetők és karbantarthatók, mivel egy-egy szolgáltatás kisebb kódbázissal rendelkezik.
- **Technológiai heterogenitás**: Különböző szolgáltatások különböző technológiákkal valósíthatók meg, ami lehetővé teszi a legmegfelelőbb eszközök használatát az adott feladatra.

**Hátrányok**:
- **Komplexitás**: A mikroszolgáltatások alapú architektúra összetettebb, mivel több szolgáltatás közötti kommunikációt és koordinációt igényel.
- **Telepítési és menedzsment nehézségek**: A sok különálló szolgáltatás kezelése és telepítése bonyolultabb lehet, különösen nagyobb rendszerek esetében.
- **Adatkezelési kihívások**: A különböző szolgáltatások saját adatbázissal rendelkeznek, ami nehezíti az adatok konzisztenciájának biztosítását és az összetett tranzakciók kezelését.

**Példa**:
Egy komplex e-kereskedelmi platform, amely különálló szolgáltatásokat tartalmaz a felhasználói menedzsmenthez, a termékkatalógushoz, a kosárkezeléshez és a rendelésfeldolgozáshoz. Minden szolgáltatás függetlenül skálázható és fejleszthető, ami lehetővé teszi az alkalmazás rugalmas és hatékony működését.

#### Rétegezett architektúra

A rétegezett architektúra egy olyan megközelítés, amelyben az alkalmazás különböző rétegekre oszlik, mindegyik réteg különálló funkciókat lát el és jól definiált interfészekkel rendelkezik. A leggyakoribb rétegek közé tartozik a prezentációs réteg, az üzleti logika réteg és az adat-hozzáférési réteg. Ez az architektúra biztosítja a modularitást és az átláthatóságot, ami megkönnyíti a fejlesztést és a karbantartást.

**Előnyök**:
- **Modularitás és újrahasznosíthatóság**: A különböző rétegek elkülönítése lehetővé teszi a modulok újrahasznosítását és független fejlesztését.
- **Karbantarthatóság**: A rétegezett struktúra átláthatóságot biztosít, ami megkönnyíti a kód karbantartását és bővítését.
- **Tesztelhetőség**: A rétegek elkülönítése lehetővé teszi az egyes rétegek különálló tesztelését, ami javítja a kód minőségét.

**Hátrányok**:
- **Teljesítmény**: A rétegek közötti kommunikáció és a különböző rétegek közötti áthaladás csökkentheti a rendszer teljesítményét.
- **Komplexitás**: Bár a rétegezés növeli az átláthatóságot, a túlzott rétegezés felesleges bonyolultságot eredményezhet.
- **Szűk keresztmetszetek**: Ha egy réteg nem skálázható megfelelően, az egész rendszer teljesítménye romolhat.

**Példa**:
Egy vállalati alkalmazás, amely három rétegre oszlik: a prezentációs réteg (felhasználói felület), az üzleti logika réteg (üzleti szabályok és folyamatok) és az adat-hozzáférési réteg (adatbázis-kezelés). Az üzleti logika réteg felelős az üzleti szabályok érvényesítéséért, míg az adat-hozzáférési réteg az adatok mentéséért és lekérdezéséért. Ez a struktúra biztosítja a modularitást és a könnyű karbantarthatóságot, ugyanakkor megfelelő szintű teljesítményt nyújt a legtöbb vállalati alkalmazás számára.

#### Következtetés

Az architektúratípusok kiválasztása és megértése alapvető fontosságú a sikeres szoftverfejlesztési projektek megvalósításához. A monolitikus architektúra egyszerűsége és könnyű megvalósíthatósága miatt kisebb projektekhez ideális, míg a mikroszolgáltatások alapú architektúra nagyobb rugalmasságot és skálázhatóságot biztosít komplex rendszerek számára. A rétegezett architektúra biztosítja a modularitást és az átláthatóságot, ami megkönnyíti a fejlesztést és a karbantartást. Az egyes architektúratípusok előnyeinek és hátrányainak figyelembevételével a fejlesztők és a tervezők képesek kiválasztani a projekt specifikus igényeinek leginkább megfelelő megközelítést.

### Architektúrális minták (MVC, MVP, MVVM)

Az architektúrális minták a szoftvertervezésben olyan bevált megoldások, amelyek segítenek a fejlesztőknek hatékonyan strukturálni és szervezni az alkalmazásokat. Az architektúrális minták használata lehetővé teszi a kód újrahasznosíthatóságát, karbantarthatóságát és skálázhatóságát. Ebben az alfejezetben három népszerű architektúrális mintát tárgyalunk részletesen: az MVC-t (Model-View-Controller), az MVP-t (Model-View-Presenter) és az MVVM-et (Model-View-ViewModel). Mindhárom minta célja a felhasználói felület és az üzleti logika szétválasztása, de mindegyik sajátos módon valósítja meg ezt a célt.

#### Model-View-Controller (MVC)

Az MVC egy klasszikus architektúrális minta, amely három fő komponenst definiál: a modellt, a nézetet és a kontrollert. Az MVC minta célja a felhasználói felület és az üzleti logika szétválasztása, ami elősegíti a kód modularitását és karbantarthatóságát.

- **Model (Modell)**: A modell a rendszer adatstruktúráit és üzleti logikáját tartalmazza. Ez felelős az adatok kezeléséért, az üzleti szabályok végrehajtásáért és az adatbázis műveletekért.
- **View (Nézet)**: A nézet a felhasználói felületért felelős. Ez jeleníti meg az adatokat a felhasználónak, és kezeli a felhasználói interakciókat.
- **Controller (Kontroller)**: A kontroller a modell és a nézet közötti közvetítő. Ez fogadja a felhasználói bemeneteket, frissíti a modellt és kiválasztja a megfelelő nézetet az adatok megjelenítéséhez.

**Példa**:
Tegyük fel, hogy egy egyszerű webes alkalmazást készítünk, amely lehetővé teszi a felhasználók számára a könyvek listájának megtekintését és új könyvek hozzáadását.

- **Model**:
    ```python
    class Book:
        def __init__(self, title, author):
            self.title = title
            self.author = author

        def save(self):
            # Kód az adatbázisba mentéshez
            pass

        @staticmethod
        def get_all():
            # Kód az összes könyv lekérdezéséhez az adatbázisból
            return []
    ```

- **View** (Django Template példa):
    ```html
    <!-- books.html -->
    <h1>Book List</h1>
    <ul>
    {% for book in books %}
        <li>{{ book.title }} by {{ book.author }}</li>
    {% endfor %}
    </ul>

    <h2>Add a new book</h2>
    <form method="post" action="/books/add/">
        <label for="title">Title:</label>
        <input type="text" id="title" name="title">
        <label for="author">Author:</label>
        <input type="text" id="author" name="author">
        <button type="submit">Add Book</button>
    </form>
    ```

- **Controller** (Django View példa):
    ```python
    from django.shortcuts import render, redirect
    from .models import Book

    def book_list(request):
        books = Book.get_all()
        return render(request, 'books.html', {'books': books})

    def add_book(request):
        if request.method == 'POST':
            title = request.POST.get('title')
            author = request.POST.get('author')
            new_book = Book(title=title, author=author)
            new_book.save()
            return redirect('/books/')
    ```

#### Model-View-Presenter (MVP)

Az MVP egy olyan architektúrális minta, amely az MVC továbbfejlesztése, és a nézet és a vezérlő közötti felelősségmegosztást célozza meg. Az MVP három fő komponenst tartalmaz: a modellt, a nézetet és a prezentert.

- **Model (Modell)**: A modell ugyanúgy működik, mint az MVC-ben, és felelős az adatok kezeléséért és az üzleti logikáért.
- **View (Nézet)**: A nézet felelős a felhasználói felület megjelenítéséért, de nem tartalmaz logikát. A nézet csak a prezentertől kapott adatokat jeleníti meg.
- **Presenter (Prezentér)**: A prezentér a közvetítő a nézet és a modell között. Ez felelős a felhasználói bemenetek kezeléséért, a modell frissítéséért és a nézet adatainak előkészítéséért.

**Példa**:
Egy egyszerű asztali alkalmazás, amely lehetővé teszi a felhasználók számára a kapcsolattartók listájának megtekintését és új kapcsolattartók hozzáadását.

- **Model**:
    ```python
    class Contact:
        def __init__(self, name, email):
            self.name = name
            self.email = email

        def save(self):
            # Kód az adatbázisba mentéshez
            pass

        @staticmethod
        def get_all():
            # Kód az összes kapcsolattartó lekérdezéséhez az adatbázisból
            return []
    ```

- **View**:
    ```python
    class ContactView:
        def display_contacts(self, contacts):
            for contact in contacts:
                print(f"{contact.name} - {contact.email}")

        def get_new_contact_info(self):
            name = input("Enter name: ")
            email = input("Enter email: ")
            return name, email
    ```

- **Presenter**:
    ```python
    class ContactPresenter:
        def __init__(self, view, model):
            self.view = view
            self.model = model

        def show_contacts(self):
            contacts = self.model.get_all()
            self.view.display_contacts(contacts)

        def add_contact(self):
            name, email = self.view.get_new_contact_info()
            new_contact = self.model(name=name, email=email)
            new_contact.save()
            self.show_contacts()
    ```

#### Model-View-ViewModel (MVVM)

Az MVVM egy architektúrális minta, amely különösen népszerű a WPF, Silverlight és más, adatközpontú felhasználói felületek fejlesztésében. Az MVVM három fő komponenst tartalmaz: a modellt, a nézetet és a nézetmodellt.

- **Model (Modell)**: A modell az adatokat és az üzleti logikát tartalmazza, ugyanúgy, mint az MVC és az MVP esetében.
- **View (Nézet)**: A nézet a felhasználói felületet képviseli, és adatkötelezést (data binding) használ a nézetmodell adataihoz való kapcsolódáshoz.
- **ViewModel (Nézetmodell)**: A nézetmodell a nézet és a modell közötti közvetítő, amely felelős az adatok előkészítéséért a nézet számára és a nézetből érkező parancsok kezeléséért. A nézetmodell biztosítja az adatkötelezést, így a nézet és a modell közötti kommunikáció minimálisra csökken.

**Példa**:
Egy WPF alkalmazás, amely lehetővé teszi a felhasználók számára a feladatok listájának megtekintését és új feladatok hozzáadását.

- **Model**:
    ```python
    class Task:
        def __init__(self, title, description):
            self.title = title
            self.description = description

        def save(self):
            # Kód az adatbázisba mentéshez
            pass

        @staticmethod
        def get_all():
            # Kód az összes feladat lekérdezéséhez az adatbázisból
            return []
    ```

- **View (XAML)**:
    ```xml
    <Window x:Class="TaskManager.MainWindow"
            xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
            xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
            Title="Task Manager" Height="350" Width="525">
        <Grid>
            <ListBox ItemsSource="{Binding Tasks}">
                <ListBox.ItemTemplate>
                    <DataTemplate>
                        <StackPanel>
                            <TextBlock Text="{Binding Title}" FontWeight="Bold"/>
                            <TextBlock Text="{Binding Description}"/>
                        </StackPanel>
                    </DataTemplate>
                </ListBox.ItemTemplate>
            </ListBox>
            <Button Content="Add Task" Command="{Binding AddTaskCommand}"/>
        </Grid>
    </Window>


    ```

- **ViewModel**:
    ```python
    import tkinter as tk
    from tkinter import simpledialog

    class TaskViewModel:
        def __init__(self, model):
            self.model = model
            self.tasks = self.model.get_all()

        def add_task(self):
            title = simpledialog.askstring("Input", "Enter task title:")
            description = simpledialog.askstring("Input", "Enter task description:")
            if title and description:
                new_task = self.model(title=title, description=description)
                new_task.save()
                self.tasks.append(new_task)
    ```

#### Következtetés

Az MVC, MVP és MVVM architektúrális minták mindegyike különböző módon segíti a felhasználói felület és az üzleti logika szétválasztását, elősegítve a kód modularitását, karbantarthatóságát és újrahasznosíthatóságát. Az MVC egy egyszerűbb és klasszikus megközelítést kínál, míg az MVP jobban elkülöníti a nézetet és a prezentert, növelve ezzel a tesztelhetőséget és a karbantarthatóságot. Az MVVM pedig különösen hasznos az adatközpontú alkalmazások esetében, ahol az adatkötelezés és a parancsok közvetlenül a nézetmodellből származnak. A megfelelő architektúrális minta kiválasztása alapvető fontosságú a projekt specifikus igényeinek és követelményeinek megfelelően, biztosítva a hatékony és skálázható szoftvermegoldások kialakítását.
