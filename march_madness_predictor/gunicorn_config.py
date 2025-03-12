import os

workers = int(os.environ.get('GUNICORN_PROCESSES', '2'))
threads = int(os.environ.get('GUNICORN_THREADS', '4'))

forwarded_allow_ips = '*'
secure_scheme_headers = {'X-Forwarded-Proto': 'https'}

# Only listen on IPv4 (default is IPv6 and IPv4)
bind = "0.0.0.0:" + os.environ.get("PORT", "8050") 