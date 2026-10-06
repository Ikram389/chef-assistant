from chef_agent import agent
from multimodal import image_to_text, audio_to_text


THREAD_ID = "my-kitchen"


def ask_agent(message: str):

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": THREAD_ID
            }
        },
    )

    return result["messages"][-1].content


def main():

     
    print("ChefMate - Your Personal Chef")
     

    print("""
Commands:

/image path/to/image.jpg
/audio path/to/audio.m4a
/quit
""")

    while True:

        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("ChefMate: Goodbye!")
            break

         
        # IMAGE
  

        if user_input.startswith("/image "):

            image_path = user_input[7:].strip()

            try:

                print("\n Looking at your kitchen...")

                ingredients = image_to_text(
                    image_path
                )

                print("\nDetected ingredients:")
                print(ingredients)

                message = f"""
The user uploaded a picture of their kitchen.

The vision model identified these ingredients:

{ingredients}

Treat these as ingredients currently available
to the user.

Help the user decide what they can cook.
If appropriate, search the web for recipes.
"""

                answer = ask_agent(message)

                print("\nChefMate:")
                print(answer)

            except Exception as e:

                print(f"\n Image error: {e}")

            continue

        # -----------------------------------------
        # AUDIO
        # -----------------------------------------

        if user_input.startswith("/audio "):

            audio_path = user_input[7:].strip()

            try:

                print("\n Transcribing your voice...")

                transcript = audio_to_text(
                    audio_path
                )

                print(f"\nYou said:\n{transcript}")

                answer = ask_agent(transcript)

                print("\nChefMate:")
                print(answer)

            except Exception as e:

                print(f"\n Audio error: {e}")

            continue

        # NORMAL TEXT
  

        try:

            answer = ask_agent(
                user_input
            )

            print("\nChefMate:")
            print(answer)

        except Exception as e:

            print(f"\n Error: {e}")


if __name__ == "__main__":
    main()