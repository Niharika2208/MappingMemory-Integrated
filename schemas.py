# Define the schemas
person_schema = {
    "Lebewesen": {
        "description": "A general entity with no specific type.",
        "attributes": ["Name", "Typ", "Alter"],
        "children": ["Person", "Haustier"]
    },
    "Person": {
        "description": "A person, can be a pupil or teacher.",
        "attributes": ["Name", "Typ", "Alter", "Rolle"],
        "children": ["Student", "Lehrer"]
    },
    "Student": {
        "description": "A person who is a pupil.",
        "attributes": ["Name", "Typ", "Alter", "Durchschnitt"],
        "children": []
    },
    "Lehrer": {
        "description": "A person who is a teacher.",
        "attributes": ["Name", "Typ", "Alter", "Rolle", "Fach"],
        "children": []
    },
    "Haustier": {
        "description": "An entity that is a pet.",
        "attributes": ["Name", "Typ", "Alter", "Spezies"],
        "children": []
    }
}


manufacturing_schema = {

}