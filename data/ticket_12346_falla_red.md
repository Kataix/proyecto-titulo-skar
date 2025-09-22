Solución Ticket 12346: Intermitencia en red de piso 3
Síntoma: Usuarios del piso 3, conectados a la red cableada, reportan pérdida de paquetes y lentitud intermitente al acceder a recursos internos. La red WiFi no presenta problemas.

Diagnóstico:

Se confirmó que los equipos afectados pertenecen a la VLAN 30 (IPs en el rango 192.168.30.0/24).

Se realizaron pruebas de ping continuo desde un equipo afectado hacia el gateway (192.168.30.1), mostrando una pérdida de paquetes del 15%.

El switch principal del piso, SW-CORP-P3-01, no reporta errores en sus logs. La configuración de la VLAN 30 es correcta.

Solución Aplicada:
Se realizó una revisión física del cableado en el rack del piso 3. Se detectó que el puerto 22 del patch panel, que conecta a una de las áreas más afectadas, tenía una conexión floja. Se volvió a crimpear el cable y se certificó el puerto. La pérdida de paquetes cesó inmediatamente.

Causa Raíz: Conexión física defectuosa en el patch panel.