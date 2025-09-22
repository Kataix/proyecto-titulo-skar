Solución Ticket 12345: Error 503 en Portal de Clientes
Síntoma: Múltiples usuarios reportan un error 503 Service Unavailable al intentar acceder al portal de clientes.
Diagnóstico: El análisis de logs en srv-web-01 mostró que el pool de conexiones a la base de datos db-clientes-prod estaba saturado.
Solución Aplicada: Se realizó un reinicio controlado del servicio Apache en srv-web-01 siguiendo el procedimiento estándar. Esto liberó las conexiones bloqueadas y restauró el servicio.
Causa Raíz: Aumento inesperado del tráfico debido a campaña de marketing. Se recomienda monitorear y escalar los recursos del pool de conexiones.