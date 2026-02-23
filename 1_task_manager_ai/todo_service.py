# todo_service.py

tasks = []
id_counter = 1

def get_tasks():
    return tasks

def add_task(title):
    global id_counter
    new_task = {
        "id": id_counter,
        "title": title,
        "status": "בטיפול"
    }
    tasks.append(new_task)
    id_counter += 1
    return new_task