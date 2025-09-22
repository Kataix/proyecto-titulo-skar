Guía Rápida: Solución a Error 'Acceso Denegado' en Módulo de Finanzas ERP
Aplicación: Sistema ERP Interno
Error Común: El usuario reporta el mensaje "Error: Acceso denegado al módulo de Finanzas" al intentar abrir la sección de contabilidad.

Causa y Solución
Este error ocurre cuando el usuario no pertenece al grupo de seguridad correcto en el Active Directory. El módulo de Finanzas requiere que el usuario sea miembro del grupo GRP-ERP-FINANZAS.

Pasos para la Solución:

Abrir la consola de "Usuarios y equipos de Active Directory".

Buscar al usuario afectado.

Hacer clic derecho sobre el usuario y seleccionar "Propiedades".

Ir a la pestaña "Miembro de".

Verificar si el grupo GRP-ERP-FINANZAS está en la lista.

Si no está, hacer clic en "Agregar...", escribir el nombre del grupo y aceptarlo.

Pedir al usuario que cierre sesión en su equipo y vuelva a iniciarla para que los cambios de permisos se apliquen.