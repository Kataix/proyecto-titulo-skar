Procedimiento de Reinicio del Servidor Web Apache
Responsable: Analista de Operaciones Nivel 1
Fecha de última actualización: 2025-08-15

Pasos para el Reinicio Controlado
Notificar al equipo: Antes de proceder, notificar en el canal de Slack #operaciones que se realizará un reinicio del servidor srv-web-01.

Conexión SSH: Conectarse al servidor utilizando las credenciales estándar: ssh analista@srv-web-01.

Verificar estado del servicio: Ejecutar el comando sudo systemctl status apache2 para confirmar el estado actual del servicio.

Ejecutar reinicio: Utilizar el comando sudo systemctl restart apache2.

Validación final: Volver a ejecutar sudo systemctl status apache2 para asegurar que el servicio esté active (running).