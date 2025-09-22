from skar_api.pipeline import rag_chain

def main():
    print("Bienvenido a SKAR, tu agente de IA para conocimiento operativo.")
    print("Escribe 'salir' para terminar la conversación.")

    while True:
        query = input("\nPregunta a SKAR: ")

        if query.lower() == 'salir':
            print("Hasta luego.")
            break

        if not query.strip():
            continue

        print("\nProcesando...")

        # Invocar la cadena de RAG
        response = rag_chain.invoke(query)

        print("\nRespuesta de SKAR:")
        print(response)

if __name__ == "__main__":
    main()