from django.urls import path
from .views import identify_plant  # identify_plantビューをインポート

urlpatterns = [
    path('identify/', identify_plant, name='identify_plant'),  # URLパターンの追加
]
