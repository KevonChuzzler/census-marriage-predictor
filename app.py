app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, leave as "*". Later, change to your Vercel URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)