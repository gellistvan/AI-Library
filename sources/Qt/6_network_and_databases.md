
## Chapter 6: Networking and Databases 


Chapter 6 of your Qt programming course, titled "Networking and Databases," covers the essential techniques and tools for building networked applications and handling data persistence through databases and data formats. This chapter will provide detailed insights into Qt's capabilities for network programming, writing network applications, interfacing with SQL databases, and working with XML and JSON data.

### 6.1: Network Programming (QTcpSocket, QUdpSocket)

**Overview**
Qt provides powerful classes to handle TCP and UDP communications, enabling the development of both client and server applications.
- **QTcpSocket:** Oﬀers functionalities to connect to a server, send, and receive data over TCP.
- **QUdpSocket:** Allows sending and receiving datagrams over UDP.

**Example:** A simple TCP client using `QTcpSocket`.

```cpp
#include <QTcpSocket>
#include <QCoreApplication>
#include <QDebug> 
 
int main(int argc, char *argv[]) { 
    QCoreApplication app(argc, argv); 
    QTcpSocket socket; 
 
    socket.connectToHost("example.com", 80); 
    if (socket.waitForConnected(1000)) { 
        socket.write("GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"); 
        if (socket.waitForBytesWritten(1000)) { 
            if (socket.waitForReadyRead(3000)) { 
                qDebug() << socket.readAll(); 
            } 
        } 
        socket.disconnectFromHost(); 
    } 
    return app.exec(); 
} 
```

### 6.2: Writing Network Applications

**Overview** 
Developing network applications often requires handling multiple connections, asynchronous data processing, and error management.
- **QTcpServer:** Used to create a server that can accept incoming TCP connections.
- **Signal-Slot Mechanism:** Handles asynchronous events like new connections and data receipt.

Key Concepts and Usage

* Server Setup: Conﬁguring a server to listen on a speciﬁc port.
* Client Handling: Managing multiple client connections.

**Example:** A basic TCP chat server and client using Qt networking.

```cpp
#include <QTcpServer>
#include <QTcpSocket>
#include <QDataStream>
#include <QCoreApplication>
#include <QDebug>

class ChatServer : public QTcpServer {
    Q_OBJECT
public:
    ChatServer(QObject *parent = nullptr) : QTcpServer(parent) {
        connect(this, &ChatServer::newConnection, this, &ChatServer::onNewConnection);
    }

    void startServer(int port) {
        if (!this->listen(QHostAddress::Any, port)) {
            qDebug() << "Server could not start!";
        } else {
            qDebug() << "Server started!";
        }
    }

private slots:
    void onNewConnection() {
        QTcpSocket *socket = this->nextPendingConnection();
        connect(socket, &QTcpSocket::readyRead, this, [this, socket]() { this->readData(socket); });
        connect(socket, &QTcpSocket::disconnected, socket, &QTcpSocket::deleteLater);
    }

    void readData(QTcpSocket *socket) {
        QDataStream in(socket);
        QString message;
        in >> message;
        qDebug() << "Received:" << message;
        broadcastMessage(message);
    }

    void broadcastMessage(const QString &message) {
        for (QTcpSocket *socket : this->findChildren<QTcpSocket *>()) {
            QDataStream out(socket);
            out << message;
        }
    }
};

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    ChatServer server;
    server.startServer(1234); // Starts the server on port 1234
    return app.exec();
}

#include "server.moc"
```

```cpp
#include <QTcpSocket>
#include <QDataStream>
#include <QCoreApplication>
#include <QTextStream>
#include <QDebug>

class ChatClient : public QObject {
    Q_OBJECT
public:
    ChatClient(const QString &host, int port, QObject *parent = nullptr) : QObject(parent) {
        connect(&socket, &QTcpSocket::readyRead, this, &ChatClient::readMessage);
        connect(&socket, &QTcpSocket::connected, this, []() {
            qDebug() << "Connected to the server!";
        });
        socket.connectToHost(host, port);
    }

    void sendMessage(const QString &message) {
        QDataStream out(&socket);
        out << message;
    }

private:
    QTcpSocket socket;

    void readMessage() {
        QDataStream in(&socket);
        QString message;
        in >> message;
        qDebug() << "Server:" << message;
    }
};

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    ChatClient client("localhost", 1234); // Connects to the server at localhost on port 1234

    QTextStream consoleInput(stdin);
    QString line;
    qDebug() << "Enter messages (Type 'quit' to exit):";
    do {
        line = consoleInput.readLine();
        if (!line.isEmpty()) {
            client.sendMessage(line);
        }
    } while (!line.trimmed().equals("quit"));

    return app.exec();
}

#include "client.moc"
```
**Running the Example**
1.  **Compile and run the server**: This will start listening on the specified port (1234 in the example).
2.  **Compile and run the client**: This will connect to the server and allow you to type messages into the console.

When you type a message in the client's console, it gets sent to the server, which then broadcasts it to all connected clients, including the sender. Messages are displayed in the console as they are received.
This example demonstrates basic TCP networking in Qt with simple chat functionality, where messages typed in one client can be seen in another, achieving a very rudimentary chat application.

### 6.3: SQL Database Access with Qt SQL

Qt SQL module provides a way to interact with databases using both SQL and Qt's model/view framework.
- **QSqlDatabase:** Represents a connection to a database.
- **QSqlQuery:** Used to execute SQL queries and navigate results.

**Example:** Connecting to an SQLite database and querying data.

```cpp
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QVariant>
#include <QDebug> 
 
int main() { 
    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE"); 
    db.setDatabaseName("example.db"); 
    if (!db.open()) { 
        qDebug() << "Error: connection with database failed"; 
    } else { 
        qDebug() << "Database: connection ok"; 
        QSqlQuery query("SELECT * FROM users"); 
        while (query.next()) { 
            QString username = query.value("username").toString(); 
            qDebug() << "Read from DB:" << username; 

        } 
    } 
} 
```

### 6.4: Using XML and JSON

XML and JSON are widely used data formats for storing and transferring structured data.
* **XML Handling:** Using `QXmlStreamReader` and `QXmlStreamWriter` for parsing and writing XML. 
* **JSON Handling:** Using `QJsonDocument`, `QJsonObject`, and `QJsonArray` for parsing and generating JSON data.


**Example:** Reading and writing JSON data.

```cpp
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDebug> 
 
int main() { 
    // Create JSON object 
    QJsonObject json; 
    json["name"] = "John Doe"; 
    json["age"] = 30; 
    json["city"] = "New York"; 
 
    // Convert JSON object to string 
    QJsonDocument doc(json); 
    QString strJson(doc.toJson(QJsonDocument::Compact)); 
    qDebug() << "JSON Output:" << strJson; 
 
    // Parse JSON string 
    QJsonDocument doc2 = QJsonDocument::fromJson(strJson.toUtf8()); 
    QJsonObject obj = doc2.object(); 
    qDebug() << "Read JSON:" << obj["name"].toString(); 
} 
```

Chapter 6 equips students with a strong foundation in network and database programming, enabling them to build complex, data-driven applications with Qt. By covering TCP/UDP networking, SQL 
database integration, and handling popular data formats like XML and JSON, students gain a comprehensive understanding of back-end development in modern applications.