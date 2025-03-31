import os
import sys
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import OpenAI
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper  # Ajouter cet import
import json

from dotenv import load_dotenv
load_dotenv()

class CrewAIDocumentationExpert:
    def __init__(self, output_folder="./generated_crews"):
        """Initialise l'équipe d'agents qui utilise la documentation CrewAI pour créer des crews."""
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialiser le modèle de langage
        self.llm = OpenAI(temperature=0.5)
        
        # Créer les agents et les tâches
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        
        # Créer la crew
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process=Process.sequential,
            verbose=2
        )

    def _create_agents(self):
        """Crée les agents spécialisés."""
        # Outils communs
        search_tool = self._get_search_tool()
        
        # Agent spécialiste de la documentation CrewAI
        doc_specialist = Agent(
            role="Spécialiste de la documentation CrewAI",
            goal="Comprendre en profondeur la documentation de CrewAI et extraire les informations pertinentes",
            backstory="Vous êtes un expert qui a étudié en détail la documentation de CrewAI. "
                     "Vous connaissez parfaitement la structure des agents, tâches, outils et crews. "
                     "Votre expertise permet d'identifier rapidement les fonctionnalités pertinentes pour un cas d'usage.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[search_tool]
        )
        
        # Agent analyste des besoins
        needs_analyst = Agent(
            role="Analyste des besoins utilisateur",
            goal="Analyser et comprendre précisément les besoins de l'utilisateur pour créer une crew adaptée",
            backstory="Vous êtes un expert en analyse des besoins avec une capacité exceptionnelle "
                     "à comprendre ce que les utilisateurs veulent réellement accomplir. "
                     "Vous transformez des descriptions vagues en spécifications précises.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        # Agent architecte de crews
        crew_architect = Agent(
            role="Architecte de Crews",
            goal="Concevoir la structure optimale d'une crew pour répondre aux besoins spécifiques",
            backstory="Vous êtes un architecte créatif qui conçoit des équipes d'agents efficaces. "
                     "Vous savez comment combiner différents types d'agents et structurer leurs interactions "
                     "pour résoudre des problèmes complexes de manière optimale.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        # Agent développeur de code
        code_developer = Agent(
            role="Développeur de code CrewAI",
            goal="Générer un code Python fonctionnel et bien structuré qui implémente la crew conçue",
            backstory="Vous êtes un développeur Python expérimenté spécialisé dans l'API CrewAI. "
                     "Vous écrivez un code propre, bien documenté et facile à comprendre. "
                     "Vous respectez les meilleures pratiques de programmation.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        return {
            "doc_specialist": doc_specialist,
            "needs_analyst": needs_analyst,
            "crew_architect": crew_architect,
            "code_developer": code_developer
        }

    def _create_tasks(self):
        """Crée les tâches pour les agents."""
        # Tâche pour l'extraction de la documentation
        extract_doc = Task(
            description="Extraire les informations pertinentes de la documentation CrewAI nécessaires pour créer une crew personnalisée. "
                       "Vous devez chercher et résumer les informations sur:\n"
                       "1. Comment créer et configurer des agents (paramètres importants, bonnes pratiques)\n"
                       "2. Comment définir des tâches efficaces (structure, paramètres, assignation)\n"
                       "3. Comment configurer une crew (processus, workflow, options)\n"
                       "4. Outils disponibles et intégrations pertinentes\n"
                       "Fournissez ces informations sous forme structurée.",
            expected_output="Un document structuré résumant les éléments clés de la documentation CrewAI pertinents pour créer une crew personnalisée",
            agent=self.agents["doc_specialist"]
        )
        
        # Tâche pour l'analyse des besoins
        analyze_needs = Task(
            description="Analyser en détail le besoin utilisateur décrit et le transformer en spécifications techniques pour une crew CrewAI. "
                       "Vous devez:\n"
                       "1. Identifier le domaine et l'objectif principal\n"
                       "2. Déterminer les compétences et connaissances requises\n"
                       "3. Identifier les types d'agents nécessaires (rôles, buts, etc.)\n"
                       "4. Définir les tâches principales à accomplir\n"
                       "5. Identifier les outils et ressources nécessaires\n"
                       "Fournissez ces spécifications sous forme structurée.",
            expected_output="Un document de spécifications techniques détaillant les agents, tâches et outils nécessaires pour répondre au besoin",
            agent=self.agents["needs_analyst"],
            context=[
                "L'utilisateur a besoin d'une équipe d'agents pour accomplir une tâche spécifique.",
                "Vous devez analyser ce besoin pour le transformer en spécifications techniques."
            ]
        )
        
        # Tâche pour la conception de l'architecture
        design_crew = Task(
            description="Concevoir l'architecture complète d'une crew CrewAI basée sur les spécifications techniques. "
                       "Vous devez:\n"
                       "1. Définir précisément chaque agent (rôle, but, backstory)\n"
                       "2. Définir chaque tâche (description, output attendu, contexte)\n"
                       "3. Spécifier les outils nécessaires à chaque agent\n"
                       "4. Déterminer le processus optimal (séquentiel, hiérarchique)\n"
                       "5. Définir les flux de travail et interactions entre agents\n"
                       "Fournissez un schéma détaillé de cette architecture.",
            expected_output="Un schéma d'architecture détaillé spécifiant tous les composants de la crew et leurs interactions",
            agent=self.agents["crew_architect"],
            context=[
                "Vous disposez des spécifications techniques et des informations de la documentation.",
                "Votre tâche est de concevoir l'architecture optimale pour la crew."
            ]
        )
        
        # Tâche pour le développement du code
        develop_code = Task(
            description="Développer le code Python complet pour implémenter la crew CrewAI conçue. "
                       "Vous devez:\n"
                       "1. Créer une structure de projet claire et organisée\n"
                       "2. Implémenter chaque agent avec tous ses paramètres\n"
                       "3. Implémenter chaque tâche avec tous ses paramètres\n"
                       "4. Configurer les outils nécessaires\n"
                       "5. Assembler la crew avec le processus approprié\n"
                       "6. Ajouter des commentaires et une documentation claire\n"
                       "7. Créer un README expliquant comment utiliser le code\n"
                       "Le code doit être complet, fonctionnel et prêt à l'emploi.",
            expected_output="Un code Python complet et documenté implémentant la crew conçue, ainsi qu'un README explicatif",
            agent=self.agents["code_developer"],
            context=[
                "Vous disposez de l'architecture détaillée de la crew à implémenter.",
                "Votre tâche est de transformer cette architecture en code Python fonctionnel."
            ]
        )
        
        return {
            "extract_doc": extract_doc,
            "analyze_needs": analyze_needs,
            "design_crew": design_crew,
            "develop_code": develop_code
        }

    def _get_search_tool(self):
        """Crée un outil de recherche gratuit basé sur DuckDuckGo."""
        try:
            search = DuckDuckGoSearchAPIWrapper()
            return Tool(
                name="CrewAIDocumentation",
                func=lambda query: search.run(f"CrewAI documentation {query}"),
                description="Recherche des informations dans la documentation CrewAI. "
                           "Utilise DuckDuckGo pour trouver des détails sur l'API."
            )
        except Exception as e:
            print(f"Erreur lors de la création de l'outil de recherche: {e}")
            return Tool(
                name="CrewAIDocumentation",
                func=lambda query: "Documentation CrewAI disponible sur: https://docs.crewai.com/",
                description="Recherche des informations dans la documentation CrewAI."
            )

    def create_custom_crew(self, user_need_description):
        """Crée une crew personnalisée basée sur la description des besoins utilisateur."""
        # Fournir le contexte aux tâches
        self.tasks["analyze_needs"].context.append(f"Besoin utilisateur: {user_need_description}")
        
        # Exécuter la crew pour générer le code
        result = self.crew.kickoff()
        
        # Créer un dossier pour le projet
        project_name = self._generate_project_name(user_need_description)
        project_folder = os.path.join(self.output_folder, project_name)
        os.makedirs(project_folder, exist_ok=True)
        
        # Extraire le code du résultat et l'enregistrer
        try:
            # Essayer d'extraire le code Python du résultat
            code_blocks = self._extract_code_from_result(result)
            
            if code_blocks:
                # Enregistrer les blocs de code dans des fichiers appropriés
                for i, code in enumerate(code_blocks):
                    filename = f"main.py" if i == 0 else f"module_{i}.py"
                    with open(os.path.join(project_folder, filename), "w", encoding="utf-8") as f:
                        f.write(code)
                
                # Créer un README
                with open(os.path.join(project_folder, "README.md"), "w", encoding="utf-8") as f:
                    f.write(f"# Crew CrewAI personnalisée\n\n")
                    f.write(f"Crew générée pour le besoin: {user_need_description}\n\n")
                    f.write("## Structure du projet\n\n")
                    for i, _ in enumerate(code_blocks):
                        filename = f"main.py" if i == 0 else f"module_{i}.py"
                        f.write(f"- `{filename}`\n")
                    f.write("\n## Installation\n\n")
                    f.write("```bash\npip install crewai langchain langchain-community\n```\n\n")
                    f.write("## Utilisation\n\n")
                    f.write("```bash\npython main.py\n```\n")
            else:
                # Si aucun bloc de code n'a été trouvé, enregistrer le résultat complet
                with open(os.path.join(project_folder, "result.txt"), "w", encoding="utf-8") as f:
                    f.write(result)
                
                # Créer un README basique
                with open(os.path.join(project_folder, "README.md"), "w", encoding="utf-8") as f:
                    f.write(f"# Conception de Crew CrewAI\n\n")
                    f.write(f"Analyse pour le besoin: {user_need_description}\n\n")
                    f.write("Consultez le fichier result.txt pour les détails complets.\n")
        
        except Exception as e:
            print(f"Erreur lors de l'enregistrement des résultats: {e}")
            # En cas d'erreur, enregistrer le résultat brut
            with open(os.path.join(project_folder, "result_raw.txt"), "w", encoding="utf-8") as f:
                f.write(result)
        
        return project_folder, result

    def _generate_project_name(self, description):
        """Génère un nom de projet à partir de la description."""
        # Extraire des mots-clés de la description
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', description.lower())
        
        # Filtrer les mots communs
        common_words = {'pour', 'avec', 'dans', 'des', 'les', 'qui', 'une', 'crew', 'agent', 'creer'}
        keywords = [w for w in words if w not in common_words][:3]
        
        if not keywords:
            keywords = ['custom', 'crew']
        
        # Générer un nom de projet avec un timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"crew_{'_'.join(keywords)}_{timestamp}"

    def _extract_code_from_result(self, result):
        """Extrait les blocs de code Python du résultat."""
        import re
        # Rechercher les blocs de code Python (entre ```python et ```)
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', result, re.DOTALL)
        
        # Si aucun bloc n'est trouvé, essayer sans spécifier le langage
        if not code_blocks:
            code_blocks = re.findall(r'```\s*(.*?)\s*```', result, re.DOTALL)
        
        return code_blocks


# Fonction principale pour utiliser cet outil
def create_crew_from_need(user_need, output_folder="./generated_crews"):
    """Crée une crew personnalisée basée sur le besoin utilisateur."""
    # Configurer votre clé API OpenAI ici si nécessaire
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    print(f"Création d'une crew personnalisée pour le besoin: {user_need}")
    expert = CrewAIDocumentationExpert(output_folder=output_folder)
    project_folder, result = expert.create_custom_crew(user_need)
    
    print(f"\nCrew personnalisée générée avec succès dans: {project_folder}")
    return project_folder, result


# Code d'exemple pour utiliser ce script
if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_need = " ".join(sys.argv[1:])
    else:
        user_need = input("Décrivez le besoin pour lequel vous souhaitez créer une équipe d'agents CrewAI: ")
    
    # Créer la crew personnalisée
    project_folder, _ = create_crew_from_need(user_need)