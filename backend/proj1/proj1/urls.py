from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic import TemplateView
from django.views.static import serve
from django.conf import settings

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("accounts.urls")),

    # Always serve /media/ files regardless of DEBUG setting.
    # Django's built-in static() helper silently returns [] in production,
    # causing the React catch-all to intercept media URLs and return HTML.
    re_path(
        r'^media/(?P<path>.*)$',
        serve,
        {'document_root': settings.MEDIA_ROOT},
    ),
]

# Catch-all: serve React's index.html for any route not matched above.
# This allows React Router to handle client-side navigation and makes
# page refreshes on deep links work correctly in production.
urlpatterns += [
    re_path(r'^.*$', TemplateView.as_view(template_name='index.html')),
]