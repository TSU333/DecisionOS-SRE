from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .schema import IncidentInput, DecisionResponse

def create_app(artifact_dir):
    @asynccontextmanager
    async def lifespan(app):
        from .runtime import Engine
        try:
            app.state.engine=Engine(artifact_dir,"cpu")
            app.state.error=None
        except Exception as exc:
            app.state.engine=None
            app.state.error=type(exc).__name__+": "+str(exc)
        yield
        app.state.engine=None
    app=FastAPI(title="DecisionOS-SRE MVP",lifespan=lifespan)
    @app.get("/health")
    def health():
        return {"status":"alive","execute_remediation":False}
    @app.get("/ready")
    def ready():
        if app.state.engine is None:
            raise HTTPException(503,detail=app.state.error)
        return {"status":"ready","device":"cpu"}
    @app.post("/v1/decide",response_model=DecisionResponse)
    def decide(incident:IncidentInput):
        if app.state.engine is None:
            raise HTTPException(503,detail="Model artifact unavailable")
        try:
            return app.state.engine.decide(incident)
        except ValueError as exc:
            raise HTTPException(422,detail=str(exc)) from exc
    return app
