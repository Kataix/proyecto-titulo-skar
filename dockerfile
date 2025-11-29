# 1. Usamos una "imagen base". Es como decir "necesito un computador con Python ya instalado".
FROM python:3.11-slim

# 2. Creamos una carpeta dentro de donde vivirá la app.
WORKDIR /app

# 3. Copiamos el archivo de requerimientos.
COPY requirements.txt .

# 4. Instalamos las librerías.
# El --no-cache-dir es para que el contenedor no pese tanto con archivos basura.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Ahora copiamos TODO el código (app.py, carpetas, scripts) dentro del contenedor.
COPY . .

# 6. Le decimos al contenedor qué puerto debe dejar abierto para que entremos (Streamlit usa el 8501).
EXPOSE 8501

# 7. El comando final: ¿Qué debe hacer la caja cuando la "encendamos"?
# Ejecutar tu app de Streamlit.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]