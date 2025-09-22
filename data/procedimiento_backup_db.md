Procedimiento Operativo: Backup Manual de Base de Datos PostgreSQL
Servidor: db-clientes-prod
Base de Datos: clientes_main

Pasos para el Backup Manual
Acceso al Servidor: Conectarse al servidor de base de datos vía SSH: ssh dba@db-clientes-prod.

Navegar al Directorio de Backups: Cambiar al directorio designado para backups: cd /var/backups/postgres/.

Ejecutar el Comando de Backup: Utilizar el comando pg_dump para crear el archivo de respaldo. El nombre del archivo debe seguir el formato [nombre_db]_backup_YYYY-MM-DD.sql.

Ejemplo de comando:
pg_dump -U postgres -W -F t clientes_main > clientes_main_backup_2025-09-22.sql.tar

Verificación del Archivo: Verificar que el archivo se haya creado correctamente y que su tamaño no sea cero usando el comando ls -lh.

Notificación: Notificar al equipo de DBA en el canal #dba-notificaciones que el backup manual ha sido completado exitosamente.