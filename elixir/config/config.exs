import Config
config :nx, :default_backend, EXLA.Backend
# config :nx, :default_backend, {EXLA.Backend, client: :host}
# config :exla, :compiler_mode, :xla
