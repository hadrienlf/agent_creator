from crewai_generator import create_crew_from_need

# Définir le besoin utilisateur
besoin = "Je veux une équipe d'agents pour analyser des articles scientifiques et en extraire les informations clés"

# Créer la crew personnalisée
dossier_projet, resultat = create_crew_from_need(besoin)