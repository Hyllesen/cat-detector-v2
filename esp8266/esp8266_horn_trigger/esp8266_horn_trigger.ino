#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include "arduino_secrets.h"

// --- CONFIGURATION ---
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;
const int HORN_PIN = 5; // D1 on NodeMCU
const int HORN_DURATION_MS = 800;

// Static IP details
IPAddress local_IP(192, 168, 100, 230);
IPAddress gateway(192, 168, 100, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress primaryDNS(8, 8, 8, 8);
IPAddress secondaryDNS(8, 8, 4, 4);

ESP8266WebServer server(80);
WiFiUDP udp;

// Keep-alive variables
unsigned long previousKeepAliveMillis = 0;
const unsigned long KEEP_ALIVE_INTERVAL = 120000; // 120 seconds

void handleTrigger() {
  Serial.println("Trigger received!");
  digitalWrite(HORN_PIN, HIGH);
  server.send(200, "text/plain", "Horn Triggered");
  delay(HORN_DURATION_MS);
  digitalWrite(HORN_PIN, LOW);
}

void handleRoot() {
  server.send(200, "text/plain", "ESP8266 Horn Trigger is Ready");
}

void keepNetworkAlive() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousKeepAliveMillis >= KEEP_ALIVE_INTERVAL) {
    previousKeepAliveMillis = currentMillis;
    
    // Refresh the router's ARP cache by sending a small UDP packet
    udp.beginPacket(gateway, 9); 
    udp.write("ping");
    udp.endPacket();
    
    Serial.println("ARP keep-alive sent to gateway");
  }
}

void setup() {
  Serial.begin(115200);

  // --- STATIC IP CONFIGURATION ---
  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS)) {
    Serial.println("STA Failed to configure Static IP");
  }

  pinMode(HORN_PIN, OUTPUT);
  digitalWrite(HORN_PIN, LOW);

  // Disable sleep BEFORE connecting to stay responsive
  WiFi.mode(WIFI_STA);
  WiFi.setSleepMode(WIFI_NONE_SLEEP);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected.");
  udp.begin(8888); 

  // --- OTA SETUP ---
  ArduinoOTA.setHostname("esp8266-horn");
  ArduinoOTA.begin();

  server.on("/", handleRoot);
  server.on("/trigger", handleTrigger);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  ArduinoOTA.handle();   // Required for wireless updates to work
  server.handleClient();
  keepNetworkAlive();    // Prevents the "sleep" issue
}