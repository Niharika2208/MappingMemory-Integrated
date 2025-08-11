# Sample Data
samples = {
    1: [
        {
            "name": "Jane Doe",
            "type": "person",
            "age": 30,
            "role": "teacher"
        },
        {
            "name": "Pet Doe",
            "type": "animal",
            "age": 5,
            "species": "dog"
        },
        {
            "name": "John Doe",
            "type": "person",
            "age": 25,
            "role": ""
        },
    ],
    2: [
        {
            "name": "Max",
            "type": "pet",
            "species": "dog",
            "age": 5
        },
        {
            "name": "Bella",
            "type": "pet",
            "species": "cat",
            "age": 3
        }
    ]
}

sample_solutions = {
    1: {
        "Lehrer": {
            "rule": ['(role == "teacher")'],
            "attributes": {
                "Name": "name",
                "Alter": "age",
                "Typ": "type",
                "Rolle": "role",
                "Fach": ""
            }
        },
        "Haustier": {
            "rule": ['(species == "dog")', '(type == "animal")', '(name.str.contains("pet", regex=True))'],
            "attributes": {
                "Name": "name",
                "Alter": "age",
                "Spezies": "species",
                "Typ": "type"
            },
        },
        "Person": {
            "rule": ['(type == "person")'],
            "attributes": {
                "Name": "name",
                "Alter": "age",
                "Typ": "type",
                "Rolle": "role"
            },
        },
    },
}