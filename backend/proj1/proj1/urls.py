from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic import TemplateView

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("accounts.urls")),
]

urlpatterns += static(
    settings.MEDIA_URL,
    document_root=settings.MEDIA_ROOT,
)

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT,
    )

# Catch-all: serve React's index.html for any route not matched above.
# This allows React Router to handle client-side navigation and makes
# page refreshes on deep links work correctly in production.
urlpatterns += [
    re_path(r'^.*$', TemplateView.as_view(template_name='index.html')),
]