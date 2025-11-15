class Entry:
    def __init__(self):
        self.user_id = None
        self.text_id = None
        self.text = None
        self.timestamp = None
        self.collection_phase = None
        self.is_words = None
        self.valence = None
        self.arousal = None
        self.valence_class= None 
        self.arousal_class = None 

    def __repr__(self):
        return (
            f"Entry(user_id={self.user_id}, text_id={self.text_id}, text={self.text}, timestamp={self.timestamp}, collection_phase={self.collection_phase}, is_words={self.is_words} valence={self.valence}, arousal={self.arousal}, valence_class={self.valence_class}, arousal_class={self.arousal_class})"
        )
