from typing import Set, Dict
from fastapi import WebSocket
import asyncio
import json


class ConnectionManager:
    """
    Manages WebSocket connections grouped by experiment_id.
    """

    def __init__(self):
        # experiment_id -> set of WebSocket connections
        self._connections: Dict[int, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, experiment_id: int, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            if experiment_id not in self._connections:
                self._connections[experiment_id] = set()
            self._connections[experiment_id].add(websocket)

    async def disconnect(self, experiment_id: int, websocket: WebSocket):
        async with self._lock:
            if experiment_id in self._connections:
                self._connections[experiment_id].discard(websocket)
                if not self._connections[experiment_id]:
                    del self._connections[experiment_id]

    async def broadcast(self, experiment_id: int, message: dict):
        """Broadcast a JSON message to all connections for an experiment."""
        async with self._lock:
            connections = self._connections.get(experiment_id, set()).copy()

        dead = set()
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)

        # Clean up dead connections
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections[experiment_id].discard(ws)

    async def broadcast_log(self, experiment_id: int, line: str):
        await self.broadcast(experiment_id, {
            "type": "log",
            "data": line,
        })

    async def broadcast_metrics(self, experiment_id: int,
                                metrics: list[dict]):
        await self.broadcast(experiment_id, {
            "type": "metrics",
            "data": metrics,
        })

    async def broadcast_status(self, experiment_id: int, status: str,
                               message: str = ""):
        await self.broadcast(experiment_id, {
            "type": "status",
            "status": status,
            "message": message,
        })


# Singleton
manager = ConnectionManager()
