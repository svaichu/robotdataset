if [ "$1" = "combi" ]; then
   docker compose down
   docker compose up
   exit 0
fi

if [ "$1" = "agent" ]; then
   # docker compose only agent container
   docker compose stop agent
   docker compose rm -f agent
   docker compose run --service-ports --rm agent ${2:-}
   exit 0   
fi

if [ "$1" = "sim" ]; then
   # docker compose only sim container
   docker compose stop sim
   docker compose rm -f sim
   docker compose run --service-ports --rm sim ${2:-}
   exit 0   
fi