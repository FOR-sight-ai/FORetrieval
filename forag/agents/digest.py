from copy import deepcopy
from typing import List, Optional
from pydantic import BaseModel
from smolagents import LiteLLMModel, ChatMessage, MessageRole
from .backends.common import get_litellm_model, rate_limit_handler


class DigestEntry(BaseModel):
    index: str
    query: str
    answer: str
    id: str | None = None

    @property
    def size(self) -> int:
        return len(self.query)

    def __str__(self) -> str:
        return f"{self.query} -> {self.answer}"

    def __repr__(self) -> str:
        return f"({self.query},{self.answer})"


class Digest:
    digest: List[DigestEntry] = []
    max_entries: int
    max_size: int
    model: LiteLLMModel
    default_summary = "[EMPTY]"
    summary: str = deepcopy(default_summary)

    def __init__(self, model: LiteLLMModel, max_entries: int, max_size: int = -1):
        self.model = model
        self.max_entries = max_entries
        self.max_size = max_size

    def __iter__(self):
        return iter(self.digest)

    def __len__(self):
        return len(self.digest)

    def __getitem__(self, index: int):
        return self.digest[index]

    @property
    def size(self) -> int:
        return sum([entry.size for entry in self.digest])

    def reset(self):
        self.digest = []

    def assimilate(self, entry: DigestEntry):
        n_entries_to_delete = 0
        current_size = self.size + entry.size
        current_n_entries = len(self) + 1
        while current_size > self.max_size or current_n_entries > self.max_entries:
            n_entries_to_delete += 1
            current_size = current_size - sum(
                [e.size for e in self.digest[:n_entries_to_delete]]
            )
            current_n_entries = current_n_entries - n_entries_to_delete

        system_prompt = (
            "You are a helpful, useful and VERY precise and concise assistant. You give short, precise and meaningful answers, straight to the point."
            "Your task is to summarize a list of (queries, answers) so that the user whom you want to assist knows what they were about, before being deleted."
            "You will be given that list of queries and answers and you will return a short but precise summary of it. Just the summary, no other thing."
            "Always use the user's language."
        )
        system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        user_prompt = f"Summarize the following list of queries and answers:\n{'\n'.join(str(e) for e in self.digest[:n_entries_to_delete])}"
        user_msg = ChatMessage(role=MessageRole.USER, content=user_prompt)
        with rate_limit_handler():
            self.summary = str(self.model([system_msg, user_msg]).content)
        for _ in range(n_entries_to_delete):
            self.digest.pop(0)

    def add(self, entry: DigestEntry):
        # set the id of the entry using its index and the last id of similar entries with same index
        entry.id = (
            f"{entry.index}_{len([e for e in self.digest if e.index == entry.index])}"
        )

        if len(self) >= self.max_entries or self.size + entry.size >= self.max_size:
            self.assimilate(entry)
        else:
            self.digest.append(entry)

    def get(self, id: str) -> DigestEntry:
        for entry in self.digest:
            if entry.id == id:
                return entry
        raise ValueError(f"Digest entry with id {id} not found")

    def get_all(self, index: Optional[str] = None) -> List[DigestEntry]:
        return [
            entry
            for entry in self.digest
            if (not index) or (index and entry.index == index)
        ]

    def get_all_text(self, index: Optional[str] = None) -> str:
        return "\n".join(
            [
                str(entry)
                for entry in self.digest
                if (not index) or (index and entry.index == index)
            ]
        )

    def get_context(self, query: str, index: Optional[str] = None) -> str:
        raw_context = self.get_all(index)
        system_prompt = (
            "You are a helpful, useful and VERY precise and concise assistant. You provide the user with short, precise and meaningful context, straight to the point."
            "You will be given a new prompt and a list of old queries and their answers."
            "**Step 0: Given the new prompt, check if the user is referring to a previous answer or query. This is typically the case if you cannot figure what the user is referring to."
            "If this is the case, just return <(previous_query, previous_answer)> as context and skip to step 5.**\n"
            "**Step 1: Given the new prompt, determine relevant old queries from the list of pairs of old queries and answers. The format is: <(query, answer)>**\n"
            "**Step 2: Given the new prompt and the relevant old queries, determine the most relevant old answers.**\n"
            "**Step 3: Given the new prompt and the most relevant old answers, generate a summary to be used as context for the new query.**\n"
            "**Step 4: If none of the old queries or answers were relevant, simply return 'Context: None'.**\n"
            "**Step 5: Return the context using the following template:**\n"
            "Context: <Context>\n"
        )
        system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        if self.summary == self.default_summary:
            user_prompt = (
                f"New query: {query}\nOld queries, answers and results: {raw_context}"
            )
        else:
            user_prompt = f"Summary of previous (and discarded queries): {self.summary}\nNew query: {query}\nOld queries and answers: {raw_context}\n"
        user_msg = ChatMessage(role=MessageRole.USER, content=user_prompt)

        with rate_limit_handler():
            response = str(self.model.generate([system_msg, user_msg]).content)

        return response

    def clear(self):
        self.digest = []
        self.summary = self.default_summary

    def __repr__(self) -> str:
        if len(self.digest) == 0:
            return "[Empty]"
        else:
            return "\n".join([repr(entry) for entry in self.digest])

    @classmethod
    def get_digestor(
        cls,
        backend: str,
        model_name: str,
        max_entries: int,
        max_size: int = -1,
        **kwargs,
    ) -> "Digest":
        model = get_litellm_model(backend, model_name, **kwargs)
        digest = Digest(model, max_entries, max_size)
        return digest
