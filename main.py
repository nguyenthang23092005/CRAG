from pprint import pprint
from build_graph import app
from build_vectostores import add_new_doc
from build_graph import display_graph

# choice = int(input("Enter your choice (1: Ask a queston/ 2: Add document/ 3: Check graph /?: Exit): "))
# if choice == 1:
#     question = input("Ask anything: ")
#     inputs = {"question": question}
#     for output in app.stream(inputs):
#         for key, value in output.items():
#             pprint(f"Node '{key}':")
#             pprint(value, indent=2, width=80, depth=None)
#         pprint("\n---\n")
#     pprint(value["generation"])
# elif choice == 2:
#     text = input("Enter your text: ")
#     add_new_doc(text)
# elif choice == 3:
#     display_graph()
# else:
#     print("You exited")

question = "who is putin"
inputs = {"question": question}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
        pprint(value, indent=2, width=80, depth=None)
    pprint("\n---\n")
pprint(value["generation"])
